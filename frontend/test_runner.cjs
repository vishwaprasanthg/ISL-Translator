const { chromium } = require('playwright');

(async () => {
  console.log("Launching browser with Playwright...");
  const browser = await chromium.launch({
    headless: true, // running headless
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream'
    ]
  });

  const page = await browser.newPage();
  
  // Listen for console logs
  page.on('console', msg => {
    console.log(`[BROWSER CONSOLE] [${msg.type()}] ${msg.text()}`);
  });
  
  page.on('pageerror', err => {
    console.log(`[BROWSER ERROR] ${err.message}`);
  });

  // Go to local frontend
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
  console.log("Navigated to app");

  try {
    // Wait for the Start Camera button and click it
    await page.waitForSelector('button', { timeout: 5000 });
    const buttons = await page.$$('button');
    for (const btn of buttons) {
      const text = await btn.textContent();
      if (text.includes("Start Camera")) {
        console.log("Found Start Camera button, clicking...");
        await btn.click();
        break;
      }
    }
  } catch (err) {
    console.log("Could not find button, likely page crashed.", err.message);
  }

  console.log("Waiting 10 seconds for MediaPipe initialization and metrics accumulation...");
  
  await page.waitForTimeout(10000);

  console.log("Scraping metrics...");
  
  const debugText = await page.evaluate(() => {
    const panels = Array.from(document.querySelectorAll('div'));
    const debugPanel = panels.find(el => el.textContent && el.textContent.includes('Performance Pipeline') && el.textContent.includes('MediaPipe FPS'));
    if (!debugPanel) return "Debug panel not found";
    return Array.from(debugPanel.querySelectorAll('span')).map(s => s.textContent).join('\n');
  });

  console.log("--- METRICS ---");
  console.log(debugText);
  console.log("---------------");

  await browser.close();
})();
