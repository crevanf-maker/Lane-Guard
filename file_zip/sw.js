self.addEventListener('install', (event) => {
    // Immediately take over the page
    self.skipWaiting();
    console.log('Service Worker Reset: Installing...');
});

self.addEventListener('activate', (event) => {
    // Clear any old caches that might be stuck
    event.waitUntil(
        caches.keys().then((names) => {
            return Promise.all(names.map(name => caches.delete(name)));
        })
    );
    self.clients.claim();
    console.log('Service Worker Reset: Caches Cleared.');
});

self.addEventListener('fetch', (event) => {
    // Do nothing (Always go to the network)
});