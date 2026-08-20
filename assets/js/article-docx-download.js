(() => {
  const button = document.querySelector('[data-docx-download]');
  if (!button) return;

  const label = document.querySelector('[data-docx-download-label]');
  const status = document.querySelector('[data-docx-download-status]');

  button.addEventListener('click', async () => {
    const previousLabel = label?.textContent || '';
    button.disabled = true;
    if (label) label.textContent = '正在生成...';
    if (status) status.textContent = '';

    try {
      const docx = await buildCurrentArticleDocx();
      const url = URL.createObjectURL(new Blob([docx], {
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      }));
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${articleSlug()}.docx`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      if (status) status.textContent = '已开始下载';
    } catch (error) {
      console.error(error);
      if (status) status.textContent = '生成失败，请稍后重试';
    } finally {
      button.disabled = false;
      if (label) label.textContent = previousLabel || '下载 Word 文档';
    }
  });

  async function buildCurrentArticleDocx() {
    const title = cleanText(document.querySelector('.post-title')?.textContent || document.title || 'article');
    const description = cleanText(document.querySelector('.post-description')?.textContent || '');
    const content = document.querySelector('.post-content');
    if (!content) throw new Error('Article content not found');

    const builder = new DocxBuilder(title);
    builder.addTitle(title);
    if (description) builder.addSubtitle(description);
    builder.addMeta(`来源：${location.href.split('#')[0]}`);
    builder.addSpacing();
    for (const child of [...content.children]) await builder.addBlock(child);
    return builder.package();
  }

  function articleSlug() {
    const parts = location.pathname.split('/').filter(Boolean);
    const slug = parts.at(-1) || 'article';
    return decodeURIComponent(slug).replace(/[<>:"/\\|?*\u0000-\u001f]+/g, '-').slice(0, 120) || 'article';
  }

  class DocxBuilder {
    constructor(title) {
      this.title = title;
      this.body = [];
      this.relationships = [];
      this.media = [];
      this.nextRelId = 1;
      this.nextDrawingId = 1;
    }

    addTitle(text) {
      this.body.push(paragraph([run(text, { bold: true, size: 36 })], { style: 'Title' }));
    }

    addSubtitle(text) {
      this.body.push(paragraph([run(text, { color: '65708A', size: 22 })], { style: 'Subtitle' }));
    }

    addMeta(text) {
      this.body.push(paragraph([run(text, { color: '65708A', size: 18 })], { style: 'Meta' }));
    }

    addSpacing() {
      this.body.push(paragraph([], { spacingAfter: 80 }));
    }

    async addBlock(element, listLevel = 0) {
      if (!(element instanceof Element)) return;
      if (element.matches('.anchor, script, style')) return;

      const tag = element.tagName.toLowerCase();
      if (/^h[1-6]$/.test(tag)) {
        const level = Math.min(Number(tag[1]), 3);
        const text = cleanText(element.textContent || '');
        if (text) this.body.push(paragraph([run(text, { bold: true })], { style: `Heading${level}` }));
        return;
      }

      if (tag === 'p') {
        await this.addParagraph(element);
        return;
      }

      if (tag === 'pre') {
        const text = (element.innerText || element.textContent || '').replace(/\n{3,}/g, '\n\n').trimEnd();
        if (text) this.body.push(paragraph([run(text, { font: 'Consolas', size: 18 })], { style: 'CodeBlock' }));
        return;
      }

      if (tag === 'blockquote') {
        const text = cleanText(element.innerText || element.textContent || '');
        if (text) this.body.push(paragraph([run(text, { italic: true })], { style: 'Quote' }));
        return;
      }

      if (tag === 'ul' || tag === 'ol') {
        await this.addList(element, tag === 'ol' ? 2 : 1, listLevel);
        return;
      }

      if (tag === 'table') {
        this.addTable(element);
        return;
      }

      if (tag === 'img') {
        await this.addImage(element);
        return;
      }

      for (const child of [...element.children]) await this.addBlock(child, listLevel);
    }

    async addParagraph(element) {
      const runs = this.inlineRuns([...element.childNodes].filter((node) => node.nodeName.toLowerCase() !== 'img'));
      if (runs.length > 0) this.body.push(paragraph(runs));
      for (const image of element.querySelectorAll(':scope > img')) await this.addImage(image);
    }

    async addList(element, numId, level) {
      for (const item of [...element.children].filter((child) => child.tagName?.toLowerCase() === 'li')) {
        const directNodes = [...item.childNodes].filter((child) => {
          const tag = child.nodeName.toLowerCase();
          return tag !== 'ul' && tag !== 'ol';
        });
        const runs = this.inlineRuns(directNodes);
        if (runs.length > 0) this.body.push(paragraph(runs, { numId, level }));
        for (const nested of [...item.children].filter((child) => ['ul', 'ol'].includes(child.tagName.toLowerCase()))) {
          await this.addList(nested, nested.tagName.toLowerCase() === 'ol' ? 2 : 1, level + 1);
        }
      }
    }

    addTable(element) {
      const rows = [...element.querySelectorAll('tr')].map((row) => (
        [...row.children]
          .filter((cell) => ['td', 'th'].includes(cell.tagName.toLowerCase()))
          .map((cell) => ({ header: cell.tagName.toLowerCase() === 'th', text: cleanText(cell.innerText || cell.textContent || '') }))
      )).filter((row) => row.length > 0);
      if (rows.length > 0) this.body.push(table(rows));
    }

    async addImage(image) {
      try {
        const sourceUrl = new URL(image.currentSrc || image.src, location.href);
        if (sourceUrl.origin !== location.origin) return;
        const response = await fetch(sourceUrl.href);
        if (!response.ok) return;
        const blob = await response.blob();
        const contentType = normalizeImageType(blob.type, sourceUrl.pathname);
        if (!contentType) return;

        const extension = contentType === 'image/jpeg' ? '.jpeg' : `.${contentType.split('/')[1]}`;
        const fileName = `image${this.media.length + 1}${extension}`;
        const relId = this.addRelationship('http://schemas.openxmlformats.org/officeDocument/2006/relationships/image', `media/${fileName}`);
        const width = image.naturalWidth || image.width || 960;
        const height = image.naturalHeight || image.height || 540;
        const maxWidthEmu = 5_600_000;
        let widthEmu = Math.round(width * 9525);
        let heightEmu = Math.round(height * 9525);
        if (widthEmu > maxWidthEmu) {
          const scale = maxWidthEmu / widthEmu;
          widthEmu = maxWidthEmu;
          heightEmu = Math.round(heightEmu * scale);
        }

        this.media.push({ name: `word/media/${fileName}`, contentType, data: new Uint8Array(await blob.arrayBuffer()) });
        this.body.push(paragraph([
          drawing({ relId, docPrId: this.nextDrawingId++, name: image.alt || fileName, widthEmu, heightEmu })
        ], { spacingBefore: 120, spacingAfter: 120 }));
      } catch {
        const alt = image.alt ? `图片：${image.alt}` : '';
        if (alt) this.body.push(paragraph([run(alt, { italic: true, color: '65708A' })]));
      }
    }

    inlineRuns(nodes, style = {}) {
      const result = [];
      for (const node of nodes) {
        if (node.nodeType === Node.TEXT_NODE) {
          const text = normalizeInlineText(node.textContent || '');
          if (text) result.push(run(text, style));
          continue;
        }
        if (!(node instanceof Element) || node.matches('.anchor')) continue;

        const tag = node.tagName.toLowerCase();
        if (tag === 'br') {
          result.push(breakRun());
        } else if (tag === 'code') {
          result.push(...this.inlineRuns([...node.childNodes], { ...style, font: 'Consolas', color: '111827', shading: 'EDF1F7' }));
        } else if (tag === 'strong' || tag === 'b') {
          result.push(...this.inlineRuns([...node.childNodes], { ...style, bold: true }));
        } else if (tag === 'em' || tag === 'i') {
          result.push(...this.inlineRuns([...node.childNodes], { ...style, italic: true }));
        } else if (tag === 'a') {
          const text = cleanText(node.textContent || '');
          const href = node.getAttribute('href');
          if (text && href && !href.startsWith('#')) {
            result.push(hyperlink(text, this.addRelationship(
              'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',
              new URL(href, location.href).href,
              'External'
            )));
          } else if (text) {
            result.push(run(text, style));
          }
        } else if (tag !== 'img') {
          result.push(...this.inlineRuns([...node.childNodes], style));
        }
      }
      return result;
    }

    addRelationship(type, target, targetMode = '') {
      const id = `rId${this.nextRelId++}`;
      this.relationships.push({ id, type, target, targetMode });
      return id;
    }

    package() {
      return makeZip([
        { name: '[Content_Types].xml', data: contentTypes(this.media) },
        { name: '_rels/.rels', data: packageRels() },
        { name: 'docProps/core.xml', data: coreProps(this.title) },
        { name: 'docProps/app.xml', data: appProps() },
        { name: 'word/document.xml', data: documentXml(this.body) },
        { name: 'word/styles.xml', data: stylesXml() },
        { name: 'word/numbering.xml', data: numberingXml() },
        { name: 'word/_rels/document.xml.rels', data: documentRels(this.relationships) },
        ...this.media
      ]);
    }
  }

  function normalizeImageType(type, pathname) {
    if (['image/png', 'image/jpeg', 'image/gif'].includes(type)) return type;
    const extension = pathname.toLowerCase().split('.').at(-1);
    if (extension === 'png') return 'image/png';
    if (extension === 'jpg' || extension === 'jpeg') return 'image/jpeg';
    if (extension === 'gif') return 'image/gif';
    return '';
  }

  function cleanText(value) {
    return value.replace(/\s+/g, ' ').trim();
  }

  function normalizeInlineText(value) {
    return value.replace(/\s+/g, ' ');
  }

  function paragraph(runs, options = {}) {
    const style = options.style ? `<w:pStyle w:val="${escapeXml(options.style)}"/>` : '';
    const spacing = `<w:spacing w:before="${options.spacingBefore ?? 0}" w:after="${options.spacingAfter ?? 160}" w:line="360" w:lineRule="auto"/>`;
    const numbering = options.numId
      ? `<w:numPr><w:ilvl w:val="${Math.min(options.level ?? 0, 5)}"/><w:numId w:val="${options.numId}"/></w:numPr>`
      : '';
    return `<w:p><w:pPr>${style}${numbering}${spacing}</w:pPr>${runs.join('')}</w:p>`;
  }

  function run(text, options = {}) {
    const properties = [
      options.bold ? '<w:b/>' : '',
      options.italic ? '<w:i/>' : '',
      options.underline ? '<w:u w:val="single"/>' : '',
      options.color ? `<w:color w:val="${options.color}"/>` : '',
      options.size ? `<w:sz w:val="${options.size}"/>` : '',
      options.font ? `<w:rFonts w:ascii="${escapeXml(options.font)}" w:hAnsi="${escapeXml(options.font)}" w:eastAsia="${escapeXml(options.font)}"/>` : '',
      options.shading ? `<w:shd w:val="clear" w:color="auto" w:fill="${options.shading}"/>` : ''
    ].join('');
    const content = String(text).split('\n').map((line, index) => (
      `${index === 0 ? '' : '<w:br/>'}<w:t xml:space="preserve">${escapeXml(line)}</w:t>`
    )).join('');
    return `<w:r><w:rPr>${properties}</w:rPr>${content}</w:r>`;
  }

  function breakRun() {
    return '<w:r><w:br/></w:r>';
  }

  function hyperlink(text, relId) {
    return `<w:hyperlink r:id="${relId}" w:history="1">${run(text, { color: '2868D8', underline: true })}</w:hyperlink>`;
  }

  function drawing({ relId, docPrId, name, widthEmu, heightEmu }) {
    const safeName = escapeXml(name);
    return `<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${widthEmu}" cy="${heightEmu}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="${docPrId}" name="${safeName}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="${docPrId}" name="${safeName}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="${relId}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${widthEmu}" cy="${heightEmu}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>`;
  }

  function table(rows) {
    const rowXml = rows.map((row) => `<w:tr>${row.map((cell) => `<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/><w:tcMar><w:top w:w="100" w:type="dxa"/><w:left w:w="120" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tcMar><w:shd w:val="clear" w:color="auto" w:fill="${cell.header ? 'EDF1F7' : 'FFFFFF'}"/></w:tcPr>${paragraph([run(cell.text, { bold: cell.header })], { spacingAfter: 80 })}</w:tc>`).join('')}</w:tr>`).join('');
    return `<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders><w:top w:val="single" w:sz="6" w:color="DFE5EF"/><w:left w:val="single" w:sz="6" w:color="DFE5EF"/><w:bottom w:val="single" w:sz="6" w:color="DFE5EF"/><w:right w:val="single" w:sz="6" w:color="DFE5EF"/><w:insideH w:val="single" w:sz="6" w:color="DFE5EF"/><w:insideV w:val="single" w:sz="6" w:color="DFE5EF"/></w:tblBorders></w:tblPr>${rowXml}</w:tbl>`;
  }

  function documentXml(body) {
    return xml(`<?mso-application progid="Word.Document"?><w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" mc:Ignorable="w14 wp14"><w:body>${body.join('')}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>`);
  }

  function stylesXml() {
    return xml(`<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:color w:val="293552"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>${style('Title', '标题', 44, '14213D', true, 0, 180)}${style('Subtitle', '副标题', 22, '65708A', false, 0, 260)}${style('Meta', '元信息', 18, '65708A', false, 0, 220)}${style('Heading1', '标题 1', 34, '14213D', true, 420, 160)}${style('Heading2', '标题 2', 30, '14213D', true, 360, 140)}${style('Heading3', '标题 3', 26, '14213D', true, 300, 120)}${style('Quote', '引用', 22, '293552', false, 120, 180, true)}${style('CodeBlock', '代码块', 18, '111827', false, 160, 180, false, 'EDF1F7')}</w:styles>`);
  }

  function style(id, name, size, color, bold, before, after, italic = false, shading = '') {
    return `<w:style w:type="paragraph" w:styleId="${id}"><w:name w:val="${name}"/><w:pPr><w:spacing w:before="${before}" w:after="${after}" w:line="360" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Microsoft YaHei"/>${bold ? '<w:b/>' : ''}${italic ? '<w:i/>' : ''}${shading ? `<w:shd w:val="clear" w:color="auto" w:fill="${shading}"/>` : ''}<w:color w:val="${color}"/><w:sz w:val="${size}"/></w:rPr></w:style>`;
  }

  function numberingXml() {
    return xml(`<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr><w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr></w:lvl><w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="◦"/><w:pPr><w:ind w:left="1080" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum><w:abstractNum w:abstractNumId="2"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl><w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%2."/><w:pPr><w:ind w:left="1080" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="2"/></w:num></w:numbering>`);
  }

  function contentTypes(media) {
    const defaults = new Map([
      ['rels', 'application/vnd.openxmlformats-package.relationships+xml'],
      ['xml', 'application/xml']
    ]);
    for (const item of media) defaults.set(item.name.split('.').at(-1), item.contentType);
    const defaultXml = [...defaults].map(([extension, contentType]) => `<Default Extension="${extension}" ContentType="${contentType}"/>`).join('');
    return xml(`<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">${defaultXml}<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>`);
  }

  function packageRels() {
    return xml('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>');
  }

  function documentRels(relationships) {
    const base = [
      '<Relationship Id="rStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
      '<Relationship Id="rNumbering" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
    ];
    const dynamic = relationships.map((item) => `<Relationship Id="${item.id}" Type="${item.type}" Target="${escapeXml(item.target)}"${item.targetMode ? ` TargetMode="${item.targetMode}"` : ''}/>`);
    return xml(`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${base.join('')}${dynamic.join('')}</Relationships>`);
  }

  function coreProps(title) {
    const now = new Date().toISOString();
    return xml(`<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>${escapeXml(title)}</dc:title><dc:creator>Ge Zhang · 技术笔记</dc:creator><cp:lastModifiedBy>Ge Zhang · 技术笔记</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">${now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">${now}</dcterms:modified></cp:coreProperties>`);
  }

  function appProps() {
    return xml('<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Ge Zhang · 技术笔记</Application></Properties>');
  }

  function xml(body) {
    return new TextEncoder().encode(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>${body}`);
  }

  function escapeXml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  function makeZip(files) {
    const localParts = [];
    const centralParts = [];
    let offset = 0;

    for (const file of files) {
      const name = new TextEncoder().encode(file.name);
      const data = file.data instanceof Uint8Array ? file.data : new Uint8Array(file.data);
      const crc = crc32(data);
      const local = new ArrayBuffer(30);
      const localView = new DataView(local);
      localView.setUint32(0, 0x04034b50, true);
      localView.setUint16(4, 20, true);
      localView.setUint16(6, 0x0800, true);
      localView.setUint16(8, 0, true);
      localView.setUint16(10, 0, true);
      localView.setUint16(12, 0, true);
      localView.setUint32(14, crc, true);
      localView.setUint32(18, data.length, true);
      localView.setUint32(22, data.length, true);
      localView.setUint16(26, name.length, true);
      localParts.push(new Uint8Array(local), name, data);

      const central = new ArrayBuffer(46);
      const centralView = new DataView(central);
      centralView.setUint32(0, 0x02014b50, true);
      centralView.setUint16(4, 20, true);
      centralView.setUint16(6, 20, true);
      centralView.setUint16(8, 0x0800, true);
      centralView.setUint16(10, 0, true);
      centralView.setUint16(12, 0, true);
      centralView.setUint16(14, 0, true);
      centralView.setUint32(16, crc, true);
      centralView.setUint32(20, data.length, true);
      centralView.setUint32(24, data.length, true);
      centralView.setUint16(28, name.length, true);
      centralView.setUint32(42, offset, true);
      centralParts.push(new Uint8Array(central), name);
      offset += 30 + name.length + data.length;
    }

    const centralDirectory = concat(centralParts);
    const end = new ArrayBuffer(22);
    const endView = new DataView(end);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, files.length, true);
    endView.setUint16(10, files.length, true);
    endView.setUint32(12, centralDirectory.length, true);
    endView.setUint32(16, offset, true);
    return concat([...localParts, centralDirectory, new Uint8Array(end)]);
  }

  function concat(parts) {
    const total = parts.reduce((sum, part) => sum + part.length, 0);
    const output = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) {
      output.set(part, offset);
      offset += part.length;
    }
    return output;
  }

  function crc32(buffer) {
    let crc = 0xffffffff;
    for (const byte of buffer) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 0xff];
    return (crc ^ 0xffffffff) >>> 0;
  }

  const crcTable = Array.from({ length: 256 }, (_, index) => {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    return value >>> 0;
  });
})();
