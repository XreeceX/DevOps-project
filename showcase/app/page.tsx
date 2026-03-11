"use client";

import ThemeToggle from "../components/ThemeToggle";

const TECH = ["Scikit-learn", "SMOTE", "RandomForest", "Docker", "Model Evaluation"];
const ACHIEVEMENTS = [
  "Compared fraud models with and without SMOTE to handle class imbalance.",
  "Generated confusion matrices, ROC curves, and detailed reports for analysis.",
  "Packaged the pipeline for reproducible containerized execution.",
];

export default function ShowcasePage() {
  const githubUrl = process.env.NEXT_PUBLIC_GITHUB_REPO || "https://github.com/XreeceX/DevOps-project";

  return (
    <div className="min-h-screen portfolio-bg">
      {/* Nav - matches portfolio */}
      <header className="portfolio-nav sticky top-0 z-50">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <a
              href="https://reece-rodrigues.vercel.app/"
              className="flex items-center gap-2 font-display text-lg font-semibold text-zinc-900 dark:text-white transition hover:text-accent"
            >
              RR
            </a>
            <a
              href="https://reece-rodrigues.vercel.app/"
              className="flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400 transition hover:text-zinc-900 dark:hover:text-white"
            >
              ← Back to Portfolio
            </a>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-24 pt-16">
        {/* Hero - // 01 — Overview */}
        <section className="mb-24">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">// 01 — Overview</p>
          <h1 className="mt-4 font-display text-5xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-6xl">
            Fraud Detection
          </h1>
          <p className="mt-2 text-sm uppercase tracking-widest text-zinc-500 dark:text-zinc-400">
            Case Study 5 · DevOps Project
          </p>
          <p className="mt-8 max-w-2xl text-lg leading-relaxed text-zinc-700 dark:text-zinc-400">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class imbalance
            and RandomForest for robust predictions—containerized for reproducible deployment.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-zinc-300"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-10 inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-black transition hover:bg-accentMuted"
          >
            View Code →
          </a>
        </section>

        {/* // 02 — Key Achievements */}
        <section className="mb-24">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">// 02 — Key Achievements</p>
          <h2 className="mt-4 font-display text-2xl font-bold text-zinc-900 dark:text-white sm:text-3xl">What I Built</h2>
          <ul className="mt-8 space-y-4">
            {ACHIEVEMENTS.map((a, i) => (
              <li key={i} className="flex gap-4 text-zinc-700 dark:text-zinc-400">
                <span className="text-accent">▹</span>
                <span>{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 03 — Model Results */}
        <section>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">// 03 — Model Results</p>
          <h2 className="mt-4 font-display text-2xl font-bold text-zinc-900 dark:text-white sm:text-3xl">SMOTE vs Baseline</h2>
          <p className="mt-2 text-zinc-700 dark:text-zinc-400">
            RandomForest performance with and without SMOTE resampling.
          </p>

          <div className="mt-12 grid gap-8 sm:grid-cols-2">
            <VizCard title="Class Distribution (Original)" img="/original_class_dist.png" alt="Original class distribution" />
            <VizCard title="Class Distribution (After SMOTE)" img="/smote_class_dist.png" alt="SMOTE class distribution" />
          </div>
          <div className="mt-8 grid gap-8 sm:grid-cols-2">
            <VizCard title="Confusion Matrix (Without SMOTE)" img="/confusion_matrix_no_smote.png" alt="Confusion matrix without SMOTE" />
            <VizCard title="Confusion Matrix (With SMOTE)" img="/confusion_matrix.png" alt="Confusion matrix with SMOTE" />
          </div>
          <div className="mt-8 grid gap-8 sm:grid-cols-2">
            <VizCard title="ROC Curve (Without SMOTE)" img="/roc_curve_no_smote.png" alt="ROC curve without SMOTE" />
            <VizCard title="ROC Curve (With SMOTE)" img="/roc_curve.png" alt="ROC curve with SMOTE" />
          </div>
        </section>

        {/* Footer - matches portfolio */}
        <footer className="mt-24 border-t border-zinc-200 dark:border-white/10 pt-12">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="inline-flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-500 transition hover:text-zinc-900 dark:hover:text-white"
          >
            ← Back to Reece Rodrigues
          </a>
          <p className="mt-6 text-sm text-zinc-700 dark:text-zinc-600">© 2026 Reece Rodrigues. Crafted with intent.</p>
          <p className="mt-1 text-xs text-zinc-700 dark:text-zinc-600">London, UK · AI Engineer & Full-Stack Developer</p>
        </footer>
      </main>
    </div>
  );
}

function VizCard({ title, img, alt }: { title: string; img: string; alt: string }) {
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-white/[0.02]">
      <div className="border-b border-zinc-200 dark:border-white/10 px-5 py-4">
        <h3 className="font-display font-semibold text-zinc-900 dark:text-white">{title}</h3>
      </div>
      <div className="relative flex min-h-[200px] items-center justify-center bg-zinc-100 dark:bg-zinc-900/50 p-6">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="w-full h-auto object-contain"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden flex-col items-center justify-center gap-2 p-6 text-center"
          style={{ display: "none" }}
        >
          <p className="text-sm text-zinc-500">Charts not generated yet</p>
          <p className="max-w-xs text-xs text-zinc-600">
            Run <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent">python src/Fraud_Detection.py</code> then copy <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent">static/*.png</code> to <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent">showcase/public/</code>
          </p>
        </div>
      </div>
    </div>
  );
}
