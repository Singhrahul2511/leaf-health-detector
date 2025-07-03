
const cacheName = "plantai-v1";
const filesToCache = [
  "/",
  "/index",
  "/static/style.css",
  "/static/icon-192.png",
  "/static/icon-512.png"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(cacheName).then(cache => {
      return cache.addAll(filesToCache);
    })
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
