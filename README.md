## Quick Start (Docker)

1. First run the run.sh file
2. Run `docker run -p 8000:8000 fraud-detection-app`
3. View the graphs in the `static` folder

> **UI note:** the showcase site now includes a light/dark theme toggle in the top
> right corner. The default theme respects your system preference but you can
> switch manually when previewing locally.

## Showcase (Vercel Deployment)

A recruiter-friendly showcase is included for deployment to Vercel:

1. **Deploy to Vercel**: Connect this repo to [Vercel](https://vercel.com). The build will:
   - Run the fraud detection pipeline (uses demo data if `data/` is not present)
   - Generate confusion matrices, ROC curves, and metrics
   - Build a static showcase site

2. **Optional**: Set `NEXT_PUBLIC_GITHUB_REPO` in Vercel project settings to your GitHub repo URL for the "GitHub Repo" button.

3. **Local preview**: Run `npm run build` from the project root, then open `showcase/out/index.html` or run `npx serve showcase/out`.

4. **Regenerate showcase images**: If charts are missing on the deployed site, run:
   ```bash
   pip install -r requirements.txt
   python src/Fraud_Detection.py
   # Then copy static/*.png to showcase/public/
   ```
   Or on Windows: `scripts\regenerate-showcase-images.bat`
