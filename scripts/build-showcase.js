#!/usr/bin/env node
/**
 * Build script: runs Python fraud detection pipeline, copies outputs to showcase/public, then builds Next.js.
 * On Vercel, skips Python (uses pre-committed images) due to externally-managed Python environment.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const staticDir = path.join(root, "static");
const publicDir = path.join(root, "showcase", "public");
const isVercel = process.env.VERCEL === "1";

if (!isVercel) {
  console.log("📦 Step 1: Installing Python dependencies...");
  execSync("pip install -r requirements.txt", {
    cwd: root,
    stdio: "inherit",
  });

  console.log("🔬 Step 2: Running fraud detection pipeline...");
  execSync("python src/Fraud_Detection.py", {
    cwd: root,
    stdio: "inherit",
  });

  console.log("📁 Step 3: Copying generated assets to showcase...");
  if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });
  const files = fs.readdirSync(staticDir);
  for (const f of files) {
    if (f.endsWith(".png") || f.endsWith(".csv")) {
      fs.copyFileSync(path.join(staticDir, f), path.join(publicDir, f));
    }
  }
} else {
  console.log("⏭️  Vercel: Skipping Python pipeline, using pre-committed assets in showcase/public/");
}

console.log("🌐 Building Next.js showcase...");
execSync("npm run build", {
  cwd: path.join(root, "showcase"),
  stdio: "inherit",
});

console.log("✅ Build complete! Output in showcase/out");
