const legacyHostname = 'blog-shf.pages.dev';
const productionHostname = 'zhangge.dev';

export function onRequest({ request, next }) {
  const url = new URL(request.url);
  if (url.hostname !== legacyHostname) return next();

  url.protocol = 'https:';
  url.hostname = productionHostname;
  url.port = '';
  return Response.redirect(url.toString(), 308);
}
