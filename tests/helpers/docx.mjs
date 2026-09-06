import fs from 'node:fs';
import { inflateRawSync } from 'node:zlib';
import { expect } from '@playwright/test';

export async function downloadDocx(page) {
  const pending = page.waitForEvent('download');
  // Surface the page's actionable export error instead of only a download timeout.
  const exportError = page.locator('[data-docx-download-status]').filter({ hasText: '生成失败' })
    .waitFor({ state: 'visible' }).then(async () => {
      throw new Error(await page.locator('[data-docx-download-status]').textContent());
    });
  const result = Promise.race([pending, exportError]);
  await page.locator('[data-docx-download]').click();
  const download = await result;
  const bytes = fs.readFileSync(await download.path());
  const end = bytes.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  expect(end).toBeGreaterThan(0);
  const count = bytes.readUInt16LE(end + 10);
  let offset = bytes.readUInt32LE(end + 16);
  const files = new Map();
  for (let index = 0; index < count; index++) {
    expect(bytes.readUInt32LE(offset)).toBe(0x02014b50);
    const method = bytes.readUInt16LE(offset + 10);
    const size = bytes.readUInt32LE(offset + 20);
    const nameLength = bytes.readUInt16LE(offset + 28);
    const extraLength = bytes.readUInt16LE(offset + 30);
    const commentLength = bytes.readUInt16LE(offset + 32);
    const local = bytes.readUInt32LE(offset + 42);
    const name = bytes.subarray(offset + 46, offset + 46 + nameLength).toString();
    const start = local + 30 + bytes.readUInt16LE(local + 26) + bytes.readUInt16LE(local + 28);
    const compressed = bytes.subarray(start, start + size);
    expect([0, 8]).toContain(method);
    files.set(name, method === 8 ? inflateRawSync(compressed) : compressed);
    offset += 46 + nameLength + extraLength + commentLength;
  }
  const xmlFiles = [...files].filter(([name]) => /\.(xml|rels)$/.test(name)).map(([name, bytes]) => [name, bytes.toString()]);
  const errors = await page.evaluate((entries) => entries.flatMap(([name, xml]) => {
    const document = new DOMParser().parseFromString(xml, 'application/xml');
    return document.querySelector('parsererror') ? [name] : [];
  }), xmlFiles);
  expect(errors).toEqual([]);
  return { files, download, document: files.get('word/document.xml').toString() };
}

export async function inspectDocx(page, files) {
  return page.evaluate(({ document, relationships }) => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(document, 'application/xml');
    const rels = parser.parseFromString(relationships, 'application/xml');
    const targets = new Map([...rels.documentElement.children].map(el => [el.getAttribute('Id'), el.getAttribute('Target')]));
    const ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main';
    const drawings = [...doc.getElementsByTagNameNS(ns, 'drawing')].map(drawing => {
      const props = drawing.getElementsByTagName('wp:docPr')[0];
      const extent = drawing.getElementsByTagName('wp:extent')[0];
      const rel = drawing.getElementsByTagName('a:blip')[0].getAttribute('r:embed');
      return { name: props.getAttribute('name'), target: `word/${targets.get(rel)}`, width: Number(extent.getAttribute('cx')), height: Number(extent.getAttribute('cy')) };
    });
    return { drawings, tables: doc.getElementsByTagNameNS(ns, 'tbl').length, listIds: [...doc.getElementsByTagNameNS(ns, 'numId')].map(el => el.getAttribute('w:val')) };
  }, { document: files.get('word/document.xml').toString(), relationships: files.get('word/_rels/document.xml.rels').toString() });
}
