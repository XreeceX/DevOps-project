## Quick Start (Docker)

1. First run the run.sh file
2. Run `docker run -p 8000:8000 fraud-detection-app`
3. View the graphs in the `static` folder

## Showcase (Vercel Deployment)

A recruiter-friendly showcase is included for deployment to Vercel:

1. **Deploy to Vercel**: Connect this repo to [Vercel](https://vercel.com). The build will:
   - Run the fraud detection pipeline (uses demo data if `data/` is not present)
   - Generate confusion matrices, ROC curves, and metrics
   - Build a static showcase site

2. **Optional**: Set `NEXT_PUBLIC_GITHUB_REPO` in Vercel project settings to your GitHub repo URL for the "GitHub Repo" button.

3. **Local preview**: Run `npm run build` from the project root, then open `showcase/out/index.html` or run `npx serve showcase/out`.
