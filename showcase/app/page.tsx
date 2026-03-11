"use client";

import ThemeToggle from "../components/ThemeToggle";

const TECH = ["Scikit-learn", "SMOTE", "RandomForest", "Docker", "Model Evaluation"];
const ACHIEVEMENTS = [
  "Compared fraud models with and without SMOTE to handle class imbalance.",
  "Generated confusion matrices, ROC curves, and detailed reports for analysis.",
  "Packaged the pipeline for reproducible containerized execution.",
];

const METRICS = [
  { label: "Fraud Cases Analyzed", value: "284,807", highlight: true },
  { label: "Imbalanced Class Ratio", value: "1:578", highlight: false },
  { label: "SMOTE Improvements", value: "+23% Recall", highlight: true },
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
        <section className="mb-32">
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
          
          {/* Tech Stack */}
          <div className="mt-8">
            <p className="text-xs font-semibold uppercase tracking-widest text-zinc-500 dark:text-zinc-500 mb-3">
              Tech Stack
            </p>
            <div className="flex flex-wrap gap-2">
              {TECH.map((t) => (
                <span
                  key={t}
                  className="rounded-full border border-accent/30 bg-accent/5 px-4 py-2 text-sm text-zinc-700 dark:text-zinc-300 transition hover:border-accent/60 hover:bg-accent/10"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-10 inline-flex items-center gap-2 rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-black transition hover:bg-accentMuted hover:shadow-lg hover:shadow-accent/20"
          >
            View Code →
          </a>
        </section>

        {/* Key Metrics */}
        <section className="mb-32">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">// 02 — Key Metrics</p>
          <h2 className="mt-4 font-display text-2xl font-bold text-zinc-900 dark:text-white sm:text-3xl">Dataset Overview</h2>
          
          <div className="mt-8 grid gap-6 sm:grid-cols-3">
            {METRICS.map((metric) => (
              <div
                key={metric.label}
                className="rounded-xl border border-zinc-200 dark:border-white/10 bg-gradient-to-br from-white/50 to-white/25 dark:from-white/[0.05] dark:to-white/[0.02] p-6 transition hover:border-accent/50 hover:from-accent/5 hover:to-accent/0"
              >
                <p className="text-sm font-semibold uppercase tracking-widest text-zinc-600 dark:text-zinc-400">
                  {metric.label}
                </p>
                <p className={`mt-3 text-3xl font-bold ${metric.highlight ? 'text-accent' : 'text-zinc-900 dark:text-white'}`}>
                  {metric.value}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* // 03 — Key Achievements */}
        <section className="mb-32">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">// 03 — Key Achievements</p>
          <h2 className="mt-4 font-display text-2xl font-bold text-zinc-900 dark:text-white sm:text-3xl">What I Built</h2>
          <ul className="mt-8 space-y-4">
            {ACHIEVEMENTS.map((a, i) => (
              <li key={i} className="flex gap-4 text-zinc-700 dark:text-zinc-400 group">
                <span className="mt-1 text-accent transition group-hover:translate-x-1">▹</span>
                <span className="transition group-hover:text-zinc-900 dark:group-hover:text-zinc-300">{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 04 — Model Results */}
        <section>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">// 04 — Model Results</p>
          <h2 className="mt-4 font-display text-2xl font-bold text-zinc-900 dark:text-white sm:text-3xl">SMOTE vs Baseline</h2>
          <p className="mt-2 text-zinc-700 dark:text-zinc-400">
            RandomForest performance with and without SMOTE resampling.
          </p>

          {/* Class Distribution Comparison */}
          <div className="mt-12">
            <h3 className="font-display text-lg font-semibold text-zinc-900 dark:text-white mb-6">Class Distribution Analysis</h3>
            <div className="grid gap-8 sm:grid-cols-2">
              <VizCard title="Original Distribution" subtitle="Severe class imbalance" img="/original_class_dist.png" alt="Original class distribution" accent={false} />
              <VizCard title="After SMOTE" subtitle="Balanced synthetic oversampling" img="/smote_class_dist.png" alt="SMOTE class distribution" accent={true} />
            </div>
          </div>

          {/* Confusion Matrices Comparison */}
          <div className="mt-16">
            <h3 className="font-display text-lg font-semibold text-zinc-900 dark:text-white mb-6">Model Predictions</h3>
            <div className="grid gap-8 sm:grid-cols-2">
              <VizCard title="Without SMOTE" subtitle="High false negatives due to imbalance" img="/confusion_matrix_no_smote.png" alt="Confusion matrix without SMOTE" accent={false} />
              <VizCard title="With SMOTE" subtitle="Improved fraud detection capability" img="/confusion_matrix.png" alt="Confusion matrix with SMOTE" accent={true} />
          </div>

          {/* ROC Curves Comparison */}
          <div className="mt-16">
            <h3 className="font-display text-lg font-semibold text-zinc-900 dark:text-white mb-6">Model Performance (AUC-ROC)</h3>
            <div className="grid gap-8 sm:grid-cols-2">
              <VizCard title="ROC Curve (Without SMOTE)" subtitle="Lower AUC due to class imbalance" img="/roc_curve_no_smote.png" alt="ROC curve without SMOTE" accent={false} />
              <VizCard title="ROC Curve (With SMOTE)" subtitle="Improved discriminative ability" img="/roc_curve.png" alt="ROC curve with SMOTE" accent={true} />
            </div>
          </div>
        </section>

        {/* Insights Section */}
        <section className="mt-32 rounded-2xl border border-accent/30 bg-gradient-to-br from-accent/5 to-accent/0 p-8 dark:from-accent/10 dark:to-accent/0">
          <h3 className="font-display text-xl font-bold text-zinc-900 dark:text-white">Key Insights</h3>
          <ul className="mt-6 space-y-3 text-zinc-700 dark:text-zinc-400">
            <li className="flex gap-3">
              <span className="text-accent font-bold mt-0.5">→</span>
              <span>SMOTE significantly improves recall for fraud detection, reducing false negatives in production.</span>
            </li>
            <li className="flex gap-3">
              <span className="text-accent font-bold mt-0.5">→</span>
              <span>Containerized deployment ensures consistency across development, testing, and production environments.</span>
            </li>
            <li className="flex gap-3">
              <span className="text-accent font-bold mt-0.5">→</span>
              <span>ROC curves demonstrate improved model discrimination when addressing class imbalance.</span>
            </li>
          </ul>
        </section>

        {/* Footer - matches portfolio */}
        <footer className="mt-32 border-t border-zinc-200 dark:border-white/10 pt-12">
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

function VizCard({ title, subtitle, img, alt, accent }: { title: string; subtitle?: string; img: string; alt: string; accent?: boolean }) {
  return (
    <div className={`group overflow-hidden rounded-xl border transition-all duration-300 ${
      accent 
        ? 'border-accent/30 bg-gradient-to-br from-white/60 to-white/40 dark:from-white/[0.08] dark:to-white/[0.03] hover:border-accent/60 hover:shadow-lg hover:shadow-accent/10 dark:hover:shadow-accent/5' 
        : 'border-zinc-200 dark:border-white/10 bg-gradient-to-br from-white/50 to-white/25 dark:from-white/[0.05] dark:to-white/[0.02] hover:border-zinc-300 dark:hover:border-white/20'
    }`}>
      <div className="border-b border-zinc-200 dark:border-white/10 px-6 py-4">
        <h3 className="font-display font-semibold text-zinc-900 dark:text-white">{title}</h3>
        {subtitle && <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{subtitle}</p>}
      </div>
      <div className="relative flex min-h-[300px] items-center justify-center bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900/30 dark:to-zinc-900/10 p-6 overflow-hidden">
        {/* Background gradient on hover */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent/0 to-accent/0 group-hover:from-accent/5 group-hover:to-accent/0 transition-all duration-300" />
        
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="relative z-10 w-full h-auto object-contain transition-transform duration-300 group-hover:scale-105"
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
