"use client";

const TECH = ["Scikit-learn", "SMOTE", "RandomForest", "Docker", "Model Evaluation"];
const ACHIEVEMENTS = [
  "Compared fraud models with and without SMOTE to handle class imbalance.",
  "Generated confusion matrices, ROC curves, and detailed reports for analysis.",
  "Packaged the pipeline for reproducible containerized execution.",
];

export default function ShowcasePage() {
  const githubUrl = process.env.NEXT_PUBLIC_GITHUB_REPO || "https://github.com/XreeceX/DevOps-project";

  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-zinc-800/80 bg-zinc-950/95 backdrop-blur-xl">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 transition hover:bg-zinc-800/50 hover:text-white"
          >
            ← Back to Portfolio
          </a>
        </div>
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-20">
        {/* // 01 — Overview */}
        <section className="relative mb-24 overflow-hidden rounded-2xl border border-zinc-800/60 bg-gradient-to-b from-zinc-900/80 to-zinc-950/50 p-8 sm:p-10">
          <div className="absolute -top-24 -right-24 h-48 w-48 rounded-full bg-cyan-500/5 blur-3xl" />
          <p className="font-mono text-xs font-medium uppercase tracking-widest text-cyan-500/90">
            // 01 — Overview
          </p>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-white sm:text-5xl">
            Devops Project
          </h1>
          <p className="mt-2 text-sm font-medium uppercase tracking-wider text-zinc-500">
            Case Study 5
          </p>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-zinc-300">
            Fraud detection app using python. Used to detect fraud bank transactions in the banking
            sector. It uses SMOTE to find imbalance in data and predict transactions.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-lg border border-zinc-600/80 bg-zinc-800/60 px-4 py-2 text-sm font-medium text-zinc-200"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-10 inline-flex items-center gap-2 rounded-lg bg-cyan-500/20 px-5 py-2.5 text-sm font-semibold text-cyan-400 ring-1 ring-cyan-500/30 transition hover:bg-cyan-500/30 hover:text-cyan-300"
          >
            View Code →
          </a>
        </section>

        {/* // 02 — Key Achievements */}
        <section className="mb-24">
          <p className="font-mono text-xs font-medium uppercase tracking-widest text-cyan-500/90">
            // 02 — Key Achievements
          </p>
          <h2 className="mt-4 text-2xl font-bold text-white">What I Built</h2>
          <ul className="mt-8 space-y-4">
            {ACHIEVEMENTS.map((a, i) => (
              <li key={i} className="flex gap-4 rounded-lg border border-zinc-800/80 bg-zinc-900/40 px-5 py-4">
                <span className="text-cyan-500">▹</span>
                <span className="text-zinc-300 leading-relaxed">{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 03 — Model Results */}
        <section className="mb-24">
          <p className="font-mono text-xs font-medium uppercase tracking-widest text-cyan-500/90">
            // 03 — Model Results
          </p>
          <h2 className="mt-4 text-2xl font-bold text-white">SMOTE vs Baseline Comparison</h2>
          <p className="mt-3 text-zinc-400">
            RandomForest performance with and without SMOTE resampling.
          </p>

          <div className="mt-12 grid gap-10 sm:grid-cols-2">
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

          <div className="mt-10 grid gap-10 sm:grid-cols-2">
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

          <div className="mt-10 grid gap-10 sm:grid-cols-2">
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
        <footer className="border-t border-zinc-800 py-12">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-zinc-500 transition hover:bg-zinc-800/50 hover:text-cyan-400"
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
    <div className="group overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/50 shadow-lg transition-all duration-300 hover:border-cyan-500/30 hover:shadow-xl">
      <div className="border-b border-zinc-800 bg-zinc-900/80 px-5 py-4">
        <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
      </div>
      <div className="relative flex aspect-[4/3] items-center justify-center bg-white p-5">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="max-h-full max-w-full object-contain"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden items-center justify-center bg-zinc-800/50 p-6 text-center"
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
