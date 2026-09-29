const puppeteer = require('puppeteer');

(async () => {
  console.log("Launching browser with fake media streams...");
  const browser = await puppeteer.launch({
    headless: "new",
    executablePath: 'C:\\Users\\91787\\.cache\\puppeteer\\chrome\\win64-152.0.7977.42\\chrome-win64\\chrome.exe',
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream'
    ]
  });

  const page = await browser.newPage();
  
  // Go to local frontend
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
  console.log("Navigated to app");

  // Wait for the Start Camera button and click it
  await page.waitForSelector('button');
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text.includes("Start Camera")) {
      console.log("Found Start Camera button, clicking...");
      await btn.click();
      break;
    }
  }

  console.log("Waiting 5 seconds for MediaPipe initialization and metrics accumulation...");
  await new Promise(r => setTimeout(r, 5000));

  console.log("Scraping metrics...");
  
  // The metrics are in a specific div in CameraFeed.jsx.
  // We can just grab all spans in the debug panel.
  const debugText = await page.evaluate(() => {
    const panels = Array.from(document.querySelectorAll('div'));
    const debugPanel = panels.find(el => el.textContent.includes('Performance Pipeline'));
    if (!debugPanel) return "Debug panel not found";
    return Array.from(debugPanel.querySelectorAll('span')).map(s => s.textContent).join('\n');
  });

  console.log("--- METRICS ---");
  console.log(debugText);
  console.log("---------------");

  await browser.close();
})();
