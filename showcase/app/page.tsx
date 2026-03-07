"use client";

import Image from "next/image";

const TECH = ["Scikit-learn", "SMOTE", "RandomForest", "Docker", "Model Evaluation"];
const ACHIEVEMENTS = [
  "Compared fraud models with and without SMOTE to handle class imbalance.",
  "Generated confusion matrices, ROC curves, and detailed reports for analysis.",
  "Packaged the pipeline for reproducible containerized execution.",
];

export default function ShowcasePage() {
  const githubUrl = process.env.NEXT_PUBLIC_GITHUB_REPO || "https://github.com/XreeceX/DevOps-project";

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-zinc-800/50 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-4">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="text-sm text-zinc-400 transition hover:text-zinc-100"
          >
            ← Back to Portfolio
          </a>
        </div>
      </nav>

      <main className="mx-auto max-w-3xl px-6 py-16">
        {/* // 01 — Overview */}
        <section className="mb-20">
          <p className="font-mono text-sm text-zinc-500">// 01 — Overview</p>
          <h1 className="mt-2 text-3xl font-semibold text-white">Devops Project</h1>
          <p className="mt-2 text-sm uppercase tracking-wider text-zinc-500">Case Study 5</p>
          <p className="mt-6 leading-relaxed text-zinc-400">
            Fraud detection app using python. Used to Detect fraud bank transactions in the banking
            sector. It uses SMOTE to find imbalance in data and predict transactions.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-md border border-zinc-700 bg-zinc-800/50 px-3 py-1 text-sm text-zinc-300"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-8 inline-flex items-center gap-1.5 text-sm font-medium text-zinc-300 transition hover:text-white"
          >
            View Code →
          </a>
        </section>

        {/* // 02 — Key Achievements */}
        <section className="mb-20">
          <p className="font-mono text-sm text-zinc-500">// 02 — Key Achievements</p>
          <h2 className="mt-2 text-xl font-semibold text-white">What I Built</h2>
          <ul className="mt-6 space-y-3">
            {ACHIEVEMENTS.map((a, i) => (
              <li key={i} className="flex gap-3 text-zinc-400">
                <span className="text-zinc-500">▹</span>
                <span>{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 03 — Model Results */}
        <section className="mb-20">
          <p className="font-mono text-sm text-zinc-500">// 03 — Model Results</p>
          <h2 className="mt-2 text-xl font-semibold text-white">SMOTE vs Baseline Comparison</h2>
          <p className="mt-2 text-zinc-400">
            RandomForest performance with and without SMOTE resampling.
          </p>

          <div className="mt-10 grid gap-8 sm:grid-cols-2">
            <VizCard
              title="Class Distribution (Original)"
              img="/original_class_dist.png"
              alt="Original class distribution"
            />
            <VizCard
              title="Class Distribution (After SMOTE)"
              img="/smote_class_dist.png"
              alt="SMOTE class distribution"
            />
          </div>

          <div className="mt-8 grid gap-8 sm:grid-cols-2">
            <VizCard
              title="Confusion Matrix (Without SMOTE)"
              img="/confusion_matrix_no_smote.png"
              alt="Confusion matrix without SMOTE"
            />
            <VizCard
              title="Confusion Matrix (With SMOTE)"
              img="/confusion_matrix.png"
              alt="Confusion matrix with SMOTE"
            />
          </div>

          <div className="mt-8 grid gap-8 sm:grid-cols-2">
            <VizCard
              title="ROC Curve (Without SMOTE)"
              img="/roc_curve_no_smote.png"
              alt="ROC curve without SMOTE"
            />
            <VizCard
              title="ROC Curve (With SMOTE)"
              img="/roc_curve.png"
              alt="ROC curve with SMOTE"
            />
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-zinc-800 pt-10">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="text-sm text-zinc-500 transition hover:text-zinc-300"
          >
            ← Back to Reece Rodrigues
          </a>
        </footer>
      </main>
    </div>
  );
}

function VizCard({ title, img, alt }: { title: string; img: string; alt: string }) {
  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/50">
      <div className="border-b border-zinc-800 px-4 py-3">
        <h3 className="text-sm font-medium text-zinc-200">{title}</h3>
      </div>
      <div className="relative aspect-[4/3] bg-zinc-800/50 p-4">
        <Image
          src={img}
          alt={alt}
          fill
          className="object-contain"
          unoptimized
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden items-center justify-center p-6 text-center"
          style={{ display: "none" }}
        >
          <p className="text-sm text-zinc-500">
            Run <code className="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs">python src/Fraud_Detection.py</code> then copy <code className="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs">static/*.png</code> to <code className="rounded bg-zinc-700 px-1.5 py-0.5 font-mono text-xs">showcase/public/</code>
          </p>
        </div>
      </div>
    </div>
  );
}
