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
      <header className="relative overflow-hidden border-b border-zinc-200 bg-white">
        <div className="absolute inset-0 bg-[linear-gradient(135deg,#0ea5e9_0%,transparent_50%),linear-gradient(225deg,#06b6d4_0%,transparent_50%)] opacity-[0.04]" />
        <div className="relative mx-auto max-w-4xl px-6 py-16 sm:py-20">
          <span className="inline-block rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-sky-700">
            Case Study 5
          </span>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-zinc-900 sm:text-5xl">
            Fraud Detection
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-zinc-600">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class
            imbalance and RandomForest for prediction—with confusion matrices, ROC curves, and
            containerized execution.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-lg bg-zinc-100 px-3.5 py-1.5 text-sm font-medium text-zinc-700"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={process.env.NEXT_PUBLIC_GITHUB_REPO || "https://github.com"}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-10 inline-flex items-center gap-2 rounded-lg bg-zinc-900 px-6 py-3 font-semibold text-white shadow-sm transition hover:bg-zinc-800"
          >
            View on GitHub
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      </header>

      {/* Achievements */}
      <section className="border-b border-zinc-200 bg-white px-6 py-14">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-xl font-semibold text-zinc-900">Key Achievements</h2>
          <ul className="mt-6 space-y-4">
            {ACHIEVEMENTS.map((a, i) => (
              <li key={i} className="flex gap-4">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sky-100 text-sm font-semibold text-sky-600">
                  {i + 1}
                </span>
                <span className="text-zinc-600 leading-relaxed">{a}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Visualizations */}
      <section className="bg-zinc-50 px-6 py-14">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-xl font-semibold text-zinc-900">Model Results</h2>
          <p className="mt-2 text-zinc-600">
            Comparison of RandomForest performance with and without SMOTE resampling.
          </p>

          <div className="mt-10 grid gap-8 sm:grid-cols-2">
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
      <footer className="border-t border-zinc-200 bg-white py-8 text-center text-sm text-zinc-500">
        Fraud Detection Pipeline • SMOTE + RandomForest • Containerized with Docker
      </footer>
    </div>
  );
}

function VizCard({ title, img, alt }: { title: string; img: string; alt: string }) {
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm transition hover:shadow-md">
      <div className="border-b border-zinc-100 px-5 py-4">
        <h3 className="font-medium text-zinc-900">{title}</h3>
      </div>
      <div className="relative aspect-[4/3] bg-zinc-50 p-4">
        <Image src={img} alt={alt} fill className="object-contain" unoptimized />
      </div>
    </div>
  );
}
