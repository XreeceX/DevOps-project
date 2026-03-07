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
    <div className="min-h-screen bg-[#0f1419] text-slate-200">
      {/* Hero */}
      <header className="border-b border-amber-900/30 bg-gradient-to-b from-amber-950/20 to-transparent">
        <div className="mx-auto max-w-5xl px-6 py-12">
          <p className="mb-2 text-sm font-medium uppercase tracking-widest text-amber-500">
            Case Study 5
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-white md:text-5xl">
            Fraud Detection
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-400">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class
            imbalance and RandomForest for prediction. Confusion matrices, ROC curves, and
            containerized execution.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-full border border-amber-700/50 bg-amber-950/30 px-4 py-1.5 text-sm text-amber-200"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={process.env.NEXT_PUBLIC_GITHUB_REPO || "https://github.com"}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-8 inline-flex items-center gap-2 rounded-lg bg-amber-600 px-6 py-3 font-medium text-white transition hover:bg-amber-500"
          >
            GitHub Repo →
          </a>
        </div>
      </header>

      {/* Achievements */}
      <section className="mx-auto max-w-5xl px-6 py-12">
        <h2 className="mb-6 text-2xl font-semibold text-white">Key Achievements</h2>
        <ul className="space-y-3">
          {ACHIEVEMENTS.map((a, i) => (
            <li key={i} className="flex gap-3">
              <span className="text-amber-500">▸</span>
              <span className="text-slate-300">{a}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* Visualizations */}
      <section className="mx-auto max-w-5xl px-6 py-12">
        <h2 className="mb-8 text-2xl font-semibold text-white">Model Results</h2>

        <div className="grid gap-10 md:grid-cols-2">
          <div className="rounded-xl border border-amber-900/40 bg-slate-900/50 p-4">
            <h3 className="mb-4 text-lg font-medium text-amber-200">
              Class Distribution (Original)
            </h3>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-800">
              <Image
                src="/original_class_dist.png"
                alt="Original class distribution"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          </div>
          <div className="rounded-xl border border-amber-900/40 bg-slate-900/50 p-4">
            <h3 className="mb-4 text-lg font-medium text-amber-200">
              Class Distribution (After SMOTE)
            </h3>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-800">
              <Image
                src="/smote_class_dist.png"
                alt="SMOTE class distribution"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          </div>
        </div>

        <div className="mt-10 grid gap-10 md:grid-cols-2">
          <div className="rounded-xl border border-amber-900/40 bg-slate-900/50 p-4">
            <h3 className="mb-4 text-lg font-medium text-amber-200">
              Confusion Matrix (Without SMOTE)
            </h3>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-800">
              <Image
                src="/confusion_matrix_no_smote.png"
                alt="Confusion matrix without SMOTE"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          </div>
          <div className="rounded-xl border border-amber-900/40 bg-slate-900/50 p-4">
            <h3 className="mb-4 text-lg font-medium text-amber-200">
              Confusion Matrix (With SMOTE)
            </h3>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-800">
              <Image
                src="/confusion_matrix.png"
                alt="Confusion matrix with SMOTE"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          </div>
        </div>

        <div className="mt-10 grid gap-10 md:grid-cols-2">
          <div className="rounded-xl border border-amber-900/40 bg-slate-900/50 p-4">
            <h3 className="mb-4 text-lg font-medium text-amber-200">
              ROC Curve (Without SMOTE)
            </h3>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-800">
              <Image
                src="/roc_curve_no_smote.png"
                alt="ROC curve without SMOTE"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          </div>
          <div className="rounded-xl border border-amber-900/40 bg-slate-900/50 p-4">
            <h3 className="mb-4 text-lg font-medium text-amber-200">
              ROC Curve (With SMOTE)
            </h3>
            <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-800">
              <Image
                src="/roc_curve.png"
                alt="ROC curve with SMOTE"
                fill
                className="object-contain"
                unoptimized
              />
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-amber-900/30 py-8 text-center text-sm text-slate-500">
        Fraud Detection Pipeline • SMOTE + RandomForest • Containerized with Docker
      </footer>
    </div>
  );
}
