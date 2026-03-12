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

      <main className="mx-auto max-w-6xl px-6 pb-24 pt-20">
        {/* Hero - // 01 — Overview */}
        <section className="mb-40 relative">
          {/* Decorative background gradient */}
          <div className="absolute -inset-x-20 -top-20 h-96 bg-gradient-to-b from-accent/10 to-transparent rounded-full blur-3xl -z-10 dark:from-accent/5" />
          
          <div className="space-y-6">
            <div className="inline-block">
              <p className="font-mono text-xs uppercase tracking-[0.3em] text-accent font-semibold flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-accent rounded-full animate-pulse"></span>
                // 01 — Overview
              </p>
            </div>
            
            <h1 className="mt-4 font-display text-6xl lg:text-7xl font-bold tracking-tight text-zinc-900 dark:text-white leading-tight">
              Fraud<br />Detection
            </h1>
            
            <p className="text-sm uppercase tracking-widest text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
              <span className="w-8 h-px bg-accent"></span>
              Case Study 5 · DevOps Project
            </p>
            
            <p className="mt-8 max-w-2xl text-lg leading-relaxed text-zinc-700 dark:text-zinc-400">
              ML-powered fraud detection for banking transactions. Uses SMOTE to address class imbalance
              and RandomForest for robust predictions—containerized for reproducible deployment.
            </p>
            
            {/* Tech Stack with Icons */}
            <div className="mt-10 py-6">
              <p className="text-xs font-semibold uppercase tracking-widest text-zinc-500 dark:text-zinc-600 mb-4">
                ✦ Tech Stack
              </p>
              <div className="flex flex-wrap gap-3">
                {TECH.map((t, i) => (
                  <span
                    key={t}
                    className="rounded-full border border-accent/40 bg-gradient-to-r from-accent/10 to-transparent px-4 py-2 text-sm text-zinc-700 dark:text-zinc-300 transition-all duration-300 hover:border-accent/80 hover:shadow-lg hover:shadow-accent/20 hover:scale-105"
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="mt-12 flex flex-wrap gap-4">
              <a
                href={githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="group relative inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-accent to-amber-400 px-8 py-3.5 text-sm font-semibold text-black transition-all duration-300 hover:shadow-xl hover:shadow-accent/30 hover:scale-105 active:scale-95"
              >
                <span>View Code</span>
                <span className="transition-transform group-hover:translate-x-1">→</span>
              </a>
              <a
                href="#metrics"
                className="inline-flex items-center gap-2 rounded-xl border border-zinc-300 dark:border-white/20 px-8 py-3.5 text-sm font-semibold text-zinc-900 dark:text-white transition-all duration-300 hover:border-accent/60 hover:bg-accent/5 dark:hover:bg-white/5"
              >
                <span>Learn More</span>
                <span className="transition-transform">↓</span>
              </a>
            </div>
          </div>
        </section>

        {/* Key Metrics */}
        <section className="mb-40 scroll-mt-20" id="metrics">
          <div className="space-y-4 mb-12">
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-accent font-semibold flex items-center gap-2">
              <span className="inline-block w-2 h-2 bg-accent rounded-full"></span>
              // 02 — Key Metrics
            </p>
            <h2 className="font-display text-4xl lg:text-5xl font-bold text-zinc-900 dark:text-white">
              Dataset Overview
            </h2>
          </div>
          
          <div className="grid gap-6 sm:grid-cols-3">
            {METRICS.map((metric, i) => (
              <div
                key={metric.label}
                className="group relative overflow-hidden rounded-2xl transition-all duration-500"
                style={{ animationDelay: `${i * 100}ms` }}
              >
                {/* Gradient background */}
                <div className="absolute inset-0 bg-gradient-to-br from-white/60 to-white/30 dark:from-white/[0.08] dark:to-white/[0.02] group-hover:from-white/80 group-hover:to-white/50 dark:group-hover:from-accent/20 dark:group-hover:to-accent/0 transition-all duration-300" />
                
                {/* Border */}
                <div className="absolute inset-0 border border-zinc-200 dark:border-white/10 group-hover:border-accent/50 dark:group-hover:border-accent/30 transition-colors duration-300 rounded-2xl" />
                
                {/* Content */}
                <div className="relative p-6 z-10">
                  <div className="flex items-start justify-between mb-4">
                    <p className="text-sm font-semibold uppercase tracking-widest text-zinc-600 dark:text-zinc-400">
                      {metric.label}
                    </p>
                    <span className="text-lg">{metric.highlight ? '⚡' : '📊'}</span>
                  </div>
                  <p className={`text-4xl font-bold transition-colors duration-300 ${metric.highlight ? 'text-accent' : 'text-zinc-900 dark:text-white group-hover:text-accent'}`}>
                    {metric.value}
                  </p>
                  <div className={`mt-4 h-1 rounded-full ${metric.highlight ? 'bg-accent' : 'bg-zinc-300 dark:bg-zinc-700'} group-hover:w-8 transition-all duration-500`} style={{ width: metric.highlight ? '100%' : '60%' }} />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* // 03 — Key Achievements */}
        <section className="mb-40">
          <div className="space-y-4 mb-12">
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-accent font-semibold flex items-center gap-2">
              <span className="inline-block w-2 h-2 bg-accent rounded-full"></span>
              // 03 — Key Achievements
            </p>
            <h2 className="font-display text-4xl lg:text-5xl font-bold text-zinc-900 dark:text-white">
              What I Built
            </h2>
          </div>
          
          <ul className="mt-8 space-y-6">
            {ACHIEVEMENTS.map((a, i) => (
              <li key={i} className="flex gap-4 group cursor-pointer">
                <div className="mt-1 flex-shrink-0">
                  <span className="inline-flex items-center justify-center w-6 h-6 rounded-lg bg-accent/10 group-hover:bg-accent/20 transition-all duration-300">
                    <span className="text-accent font-bold">✓</span>
                  </span>
                </div>
                <span className="text-lg text-zinc-700 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-300 group-hover:translate-x-1 transition-all duration-300">
                  {a}
                </span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 04 — Model Results */}
        <section className="mb-20">
          <div className="space-y-4 mb-16">
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-accent font-semibold flex items-center gap-2">
              <span className="inline-block w-2 h-2 bg-accent rounded-full"></span>
              // 04 — Model Results
            </p>
            <h2 className="font-display text-4xl lg:text-5xl font-bold text-zinc-900 dark:text-white">
              SMOTE vs Baseline
            </h2>
            <p className="text-zinc-700 dark:text-zinc-400 text-lg">
              RandomForest performance with and without SMOTE resampling.
            </p>
          </div>

          {/* Class Distribution Comparison */}
          <div className="mb-20">
            <h3 className="font-display text-2xl font-bold text-zinc-900 dark:text-white mb-8 flex items-center gap-2">
              <span className="text-accent">📊</span> Class Distribution Analysis
            </h3>
            <div className="grid gap-8 sm:grid-cols-2">
              <VizCard title="Original Distribution" subtitle="Severe class imbalance" img="/original_class_dist.png" alt="Original class distribution" accent={false} Icon="⚠️" />
              <VizCard title="After SMOTE" subtitle="Balanced synthetic oversampling" img="/smote_class_dist.png" alt="SMOTE class distribution" accent={true} Icon="✨" />
            </div>
          </div>

          {/* Confusion Matrices Comparison */}
          <div className="mb-20">
            <h3 className="font-display text-2xl font-bold text-zinc-900 dark:text-white mb-8 flex items-center gap-2">
              <span className="text-accent">🎯</span> Model Predictions
            </h3>
            <div className="grid gap-8 sm:grid-cols-2">
              <VizCard title="Without SMOTE" subtitle="High false negatives due to imbalance" img="/confusion_matrix_no_smote.png" alt="Confusion matrix without SMOTE" accent={false} Icon="❌" />
              <VizCard title="With SMOTE" subtitle="Improved fraud detection capability" img="/confusion_matrix.png" alt="Confusion matrix with SMOTE" accent={true} Icon="✅" />
            </div>
          </div>

          {/* ROC Curves Comparison */}
          <div>
            <h3 className="font-display text-2xl font-bold text-zinc-900 dark:text-white mb-8 flex items-center gap-2">
              <span className="text-accent">📈</span> Model Performance (AUC-ROC)
            </h3>
            <div className="grid gap-8 sm:grid-cols-2">
              <VizCard title="ROC Curve (Without SMOTE)" subtitle="Lower AUC due to class imbalance" img="/roc_curve_no_smote.png" alt="ROC curve without SMOTE" accent={false} Icon="📉" />
              <VizCard title="ROC Curve (With SMOTE)" subtitle="Improved discriminative ability" img="/roc_curve.png" alt="ROC curve with SMOTE" accent={true} Icon="📈" />
            </div>
          </div>
        </section>

        {/* Insights Section */}
        <section className="my-20 group">
          <div className="relative overflow-hidden rounded-3xl border border-accent/40 p-8 md:p-12 transition-all duration-500 group-hover:border-accent/80">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-accent/10 via-transparent to-accent/5 dark:from-accent/5 dark:via-transparent dark:to-accent/0 group-hover:from-accent/20 group-hover:to-accent/10 transition-all duration-300" />
            
            {/* Decorative elements */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-accent/5 rounded-full blur-3xl -z-10" />
            
            {/* Content */}
            <div className="relative z-10">
              <h3 className="font-display text-3xl font-bold text-zinc-900 dark:text-white flex items-center gap-3">
                <span className="text-2xl">💡</span>
                Key Insights
              </h3>
              <ul className="mt-8 space-y-5">
                {[
                  "SMOTE significantly improves recall for fraud detection, reducing false negatives in production.",
                  "Containerized deployment ensures consistency across development, testing, and production environments.",
                  "ROC curves demonstrate improved model discrimination when addressing class imbalance."
                ].map((insight, i) => (
                  <li key={i} className="flex gap-4 groups">
                    <span className="text-accent font-bold text-xl mt-0.5 group-hover:scale-125 transition-transform">→</span>
                    <span className="text-zinc-700 dark:text-zinc-400 text-lg leading-relaxed group-hover:text-zinc-900 dark:group-hover:text-zinc-300 transition-colors">
                      {insight}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* Footer - matches portfolio */}
        <footer className="mt-32 border-t border-zinc-200 dark:border-white/10 pt-12 pb-6">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="inline-flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-500 transition-colors hover:text-zinc-900 dark:hover:text-white"
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

function VizCard({ title, subtitle, img, alt, accent, Icon }: { title: string; subtitle?: string; img: string; alt: string; accent?: boolean; Icon?: string }) {
  return (
    <div className={`group h-full overflow-hidden rounded-2xl border backdrop-blur-sm transition-all duration-500 ${
      accent 
        ? 'border-accent/50 bg-gradient-to-br from-white/70 via-white/50 to-white/40 dark:from-white/[0.12] dark:via-white/[0.06] dark:to-white/[0.02] hover:border-accent/80 hover:shadow-2xl hover:shadow-accent/20 dark:hover:shadow-accent/10' 
        : 'border-zinc-200 dark:border-white/10 bg-gradient-to-br from-white/60 via-white/40 to-white/20 dark:from-white/[0.08] dark:via-white/[0.04] dark:to-white/[0.01] hover:border-zinc-300 dark:hover:border-white/20 hover:shadow-xl hover:shadow-black/5'
    }`}>
      <div className="border-b border-inherit px-6 py-5 backdrop-blur-sm">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="font-display font-semibold text-zinc-900 dark:text-white text-lg">{title}</h3>
            {subtitle && <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">{subtitle}</p>}
          </div>
          {Icon && <span className="text-2xl ml-4">{Icon}</span>}
        </div>
      </div>
      <div className="relative flex h-80 items-center justify-center bg-gradient-to-br from-zinc-50 via-white to-zinc-100 dark:from-zinc-900/40 dark:via-zinc-900/20 dark:to-zinc-900/10 p-6 overflow-hidden">
        {/* Enhanced background gradient on hover */}
        <div className={`absolute inset-0 transition-all duration-500 ${accent ? 'bg-gradient-to-br from-accent/5 to-transparent group-hover:from-accent/15 group-hover:to-accent/5' : 'bg-gradient-to-br from-transparent to-transparent group-hover:from-accent/5 group-hover:to-transparent'}`} />
        
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="relative z-10 w-full h-full object-contain transition-transform duration-500 group-hover:scale-110"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden flex-col items-center justify-center gap-2 p-6 text-center bg-white/50 dark:bg-black/30 backdrop-blur-sm"
          style={{ display: "none" }}
        >
          <p className="text-sm text-zinc-500">Charts not generated yet</p>
          <p className="max-w-xs text-xs text-zinc-600 dark:text-zinc-400">
            Run <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent text-xs">python src/Fraud_Detection.py</code> then copy <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent text-xs">static/*.png</code> to <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent text-xs">showcase/public/</code>
          </p>
        </div>
      </div>
    </div>
  );
}
