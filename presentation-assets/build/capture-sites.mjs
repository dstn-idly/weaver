import { chromium } from "/Users/dustinthach/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs";
import path from "node:path";
import { mkdir, writeFile } from "node:fs/promises";

const ROOT = "/Users/dustinthach/Documents/Coding Projects/Hackathon/Greptile YC/presentation-assets";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const groups = {
  dealerships: [
    ["01-nissan-of-portland", "Nissan of Portland", "https://nissanofportland.com/"],
    ["02-alpine-nissan", "Alpine Nissan", "https://alpinenissan.com/inventory"],
    ["03-teddy-nissan", "Teddy Nissan", "https://www.teddynissan.com/inventory/new"],
    ["04-town-center-nissan", "Town Center Nissan", "https://www.towncenternissan.com/new-inventory"],
    ["05-xtreme-nissan", "Xtreme Nissan", "https://www.xtremenissan.com/inventory/new"],
    ["06-lupient-nissan", "Lupient Nissan", "https://lupientnissan.com/inventory"],
    ["07-tamaroff-nissan", "Tamaroff Nissan", "https://www.tamaroffnissan.com/"],
    ["08-state-line-nissan", "State Line Nissan", "https://statelinenissan.com/"],
    ["09-777-nissan", "777 Nissan", "https://777nissan.com/cars/new-inventory"],
  ],
  ecommerce: [
    ["01-ikea", "IKEA", "https://www.ikea.com/us/en/"],
    ["02-lego", "LEGO Shop", "https://www.lego.com/en-us"],
    ["03-patagonia", "Patagonia", "https://www.patagonia.com/home/"],
    ["04-rei", "REI Co-op", "https://www.rei.com/"],
    ["05-apple", "Apple Store", "https://www.apple.com/store"],
    ["06-warby-parker", "Warby Parker", "https://www.warbyparker.com/"],
    ["07-allbirds", "Allbirds", "https://www.allbirds.com/"],
    ["08-etsy", "Etsy", "https://www.etsy.com/"],
    ["09-crate-and-barrel", "Crate & Barrel", "https://www.crateandbarrel.com/"],
  ],
};

async function dismissCommonOverlays(page) {
  const labels = [
    "Accept all cookies",
    "Accept All Cookies",
    "Accept all",
    "Accept All",
    "Allow all",
    "Allow All",
    "Accept",
    "OK",
    "Ok",
    "ok",
    "I agree",
    "Agree",
    "Got it",
    "Okay",
    "Continue",
    "Close",
    "Continue without accepting",
  ];
  for (const label of labels) {
    const button = page.getByRole("button", { name: label, exact: false }).first();
    if (await button.isVisible().catch(() => false)) {
      await button.click({ timeout: 1500 }).catch(() => {});
      await page.waitForTimeout(350);
      break;
    }
  }
  await page.keyboard.press("Escape").catch(() => {});
  const closers = page.locator('button[aria-label*="close" i], button[title*="close" i], [role="button"][aria-label*="close" i]');
  const closerCount = await closers.count().catch(() => 0);
  for (let index = closerCount - 1; index >= 0 && index >= closerCount - 3; index -= 1) {
    const closer = closers.nth(index);
    if (await closer.isVisible().catch(() => false)) {
      await closer.click({ timeout: 1000 }).catch(() => {});
      await page.waitForTimeout(180);
    }
  }
}

async function removeTransientOverlays(page) {
  const phrases = [
    "We use cookies to recognize you",
    "Sign in for the best experience",
    "Your Data, Your Choice",
    "Privacy preferences: The control is yours",
  ];
  for (const phrase of phrases) {
    const matches = await page.getByText(phrase, { exact: false }).all().catch(() => []);
    for (const match of matches) {
      if (!(await match.isVisible().catch(() => false))) continue;
      await match.evaluate((element) => {
        let node = element;
        for (let depth = 0; depth < 7 && node?.parentElement; depth += 1) {
          const style = getComputedStyle(node);
          const z = Number.parseInt(style.zIndex || "0", 10);
          if (["fixed", "sticky", "absolute"].includes(style.position) && (z > 5 || node.getBoundingClientRect().height > 140)) {
            node.remove();
            return;
          }
          node = node.parentElement;
        }
        element.remove();
      }).catch(() => {});
    }
  }
}

async function capture(browser, group, item) {
  const [slug, name, url] = item;
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 1.5,
    locale: "en-US",
    colorScheme: "light",
    userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
  });
  const page = await context.newPage();
  const output = path.join(ROOT, group, `${slug}.png`);
  const result = { group, slug, name, requestedUrl: url, output, ok: false };
  try {
    const response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 35000 });
    await page.waitForLoadState("networkidle", { timeout: 9000 }).catch(() => {});
    await page.waitForTimeout(1800);
    await dismissCommonOverlays(page);
    await page.waitForTimeout(700);
    await dismissCommonOverlays(page);
    await removeTransientOverlays(page);
    await page.evaluate(() => window.scrollTo(0, Math.min(240, document.documentElement.scrollHeight))).catch(() => {});
    await page.waitForTimeout(450);
    await page.evaluate(() => window.scrollTo(0, 0)).catch(() => {});
    await page.waitForTimeout(700);
    await page.screenshot({ path: output, type: "png", fullPage: false, animations: "disabled" });
    result.ok = true;
    result.status = response?.status();
    result.finalUrl = page.url();
    result.title = await page.title();
  } catch (error) {
    result.error = String(error?.message || error);
  } finally {
    await context.close();
  }
  process.stdout.write(`${JSON.stringify(result)}\n`);
  return result;
}

async function main() {
  const group = process.argv[2];
  if (!groups[group]) throw new Error(`Choose one group: ${Object.keys(groups).join(", ")}`);
  const slugFilter = process.argv[3];
  const selected = slugFilter ? groups[group].filter(([slug]) => slug === slugFilter) : groups[group];
  if (!selected.length) throw new Error(`No capture target matched ${slugFilter}`);
  await mkdir(path.join(ROOT, group), { recursive: true });
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const results = [];
  try {
    for (const item of selected) results.push(await capture(browser, group, item));
  } finally {
    await browser.close();
  }
  const resultsName = slugFilter ? `capture-results-${slugFilter}.json` : "capture-results.json";
  await writeFile(path.join(ROOT, group, resultsName), JSON.stringify(results, null, 2) + "\n", "utf8");
  if (results.some((item) => !item.ok)) process.exitCode = 2;
}

await main();
