/**
 * Umami event tracking. The Umami tag is injected server-side only when
 * UMAMI_DOMAIN/UMAMI_ID are set (production); everywhere else — local dev,
 * tests, ad-blocked browsers — window.umami is undefined and trackEvent is a
 * no-op. The try/catch guarantees analytics can never break the sim.
 */
export function trackEvent(name, props) {
    try {
        window.umami?.track(name, props);
    } catch { /* analytics must never break the app */ }
}
