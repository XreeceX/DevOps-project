"use client";

import Image from "next/image";

const TECH = ["Scikit-learn", "SMOTE", "RandomForest", "Docker", "Matplotlib", "Pandas"];
const ACHIEVEMENTS = [
  "Compared fraud models with and without SMOTE to handle class imbalance.",
  "Generated confusion matrices, ROC curves, and detailed reports for analysis.",
  "Packaged the pipeline for reproducible containerized execution.",
];

export default function ShowcasePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <header className="relative overflow-hidden border-b border-white/10">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 via-transparent to-cyan-500/5" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(34,197,94,0.15),transparent)]" />
        <div className="relative mx-auto max-w-5xl px-6 py-20 sm:py-28">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-widest text-emerald-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Case Study 5
          </span>
          <h1 className="mt-6 text-5xl font-bold tracking-tight text-white sm:text-6xl lg:text-7xl">
            Fraud Detection
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-zinc-400">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class
            imbalance and RandomForest for prediction—with confusion matrices, ROC curves, and
            containerized execution.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-zinc-300 backdrop-blur-sm"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={process.env.NEXT_PUBLIC_GITHUB_REPO || "https://github.com"}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-10 inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-6 py-3.5 font-semibold text-zinc-900 shadow-lg shadow-emerald-500/25 transition hover:bg-emerald-400 hover:shadow-emerald-500/30"
          >
            View on GitHub
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      </header>

      {/* Achievements */}
      <section className="border-b border-white/10 px-6 py-16">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-sm font-semibold uppercase tracking-widest text-emerald-400">
            Key Achievements
          </h2>
          <ul className="mt-8 grid gap-6 sm:grid-cols-3">
            {ACHIEVEMENTS.map((a, i) => (
              <li
                key={i}
                className="group rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm transition hover:border-emerald-500/30 hover:bg-white/[0.07]"
              >
                <span className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/20 text-lg font-bold text-emerald-400">
                  {i + 1}
                </span>
                <span className="text-zinc-300 leading-relaxed">{a}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Visualizations */}
      <section className="px-6 py-20">
        <div className="mx-auto max-w-5xl">
          <div className="mb-12">
            <h2 className="text-sm font-semibold uppercase tracking-widest text-emerald-400">
              Model Results
            </h2>
            <p className="mt-3 text-xl font-semibold text-white">
              SMOTE vs Baseline Comparison
            </p>
            <p className="mt-2 text-zinc-400">
              RandomForest performance with and without SMOTE resampling.
            </p>
          </div>

          <div className="grid gap-8 sm:grid-cols-2">
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
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-10 text-center">
        <p className="text-sm text-zinc-500">
          Fraud Detection Pipeline • SMOTE + RandomForest • Containerized with Docker
        </p>
      </footer>
    </div>
  );
}

function VizCard({ title, img, alt }: { title: string; img: string; alt: string }) {
  return (
    <div className="group overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03] shadow-xl shadow-black/20 transition-all duration-300 hover:border-emerald-500/30 hover:bg-white/[0.06] hover:shadow-emerald-500/5">
      <div className="border-b border-white/10 px-6 py-5">
        <h3 className="font-semibold text-white">{title}</h3>
      </div>
      <div className="relative aspect-[4/3] bg-zinc-50 p-6 shadow-inner">
        <Image src={img} alt={alt} fill className="object-contain" unoptimized />
      </div>
    </div>
  );
}
