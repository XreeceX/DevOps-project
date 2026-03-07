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
    <div className="relative min-h-screen bg-[#0a0a0c] noise-overlay">
      {/* Subtle grid background */}
      <div className="fixed inset-0 bg-grid-pattern bg-grid opacity-40" aria-hidden />

      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#0a0a0c]/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="group flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 transition hover:bg-white/[0.04] hover:text-white"
          >
            <span className="transition-transform group-hover:-translate-x-0.5">←</span>
            Back to Portfolio
          </a>
        </div>
      </nav>

      <main className="relative z-10 mx-auto max-w-4xl px-6 py-16 sm:py-24">
        {/* // 01 — Overview */}
        <section className="group relative mb-28 overflow-hidden rounded-2xl border border-white/[0.08] bg-gradient-to-b from-white/[0.04] to-transparent p-8 sm:p-12 transition-colors hover:border-cyan-500/20">
          <div className="absolute -top-32 -right-32 h-64 w-64 rounded-full bg-cyan-500/[0.07] blur-3xl transition-opacity group-hover:bg-cyan-500/[0.12]" />
          <div className="absolute -bottom-20 -left-20 h-40 w-40 rounded-full bg-emerald-500/[0.04] blur-2xl" />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-cyan-400/90">
            // 01 — Overview
          </p>
          <h1 className="mt-5 text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
            Fraud Detection
          </h1>
          <p className="mt-3 text-sm font-medium uppercase tracking-widest text-zinc-500">
            Case Study 5 · DevOps Project
          </p>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-zinc-400">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class imbalance
            and RandomForest for robust predictions—containerized for reproducible deployment.
          </p>
          <div className="mt-8 flex flex-wrap gap-2.5">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-full border border-white/[0.08] bg-white/[0.03] px-4 py-1.5 text-sm font-medium text-zinc-300 transition hover:border-cyan-500/30 hover:bg-cyan-500/10 hover:text-cyan-300"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="group/btn mt-10 inline-flex items-center gap-2 rounded-xl bg-cyan-500/15 px-5 py-3 text-sm font-semibold text-cyan-400 ring-1 ring-cyan-500/25 transition hover:bg-cyan-500/25 hover:ring-cyan-500/40 hover:text-cyan-300"
          >
            View Code
            <span className="transition-transform group-hover/btn:translate-x-0.5">→</span>
          </a>
        </section>

        {/* // 02 — Key Achievements */}
        <section className="mb-28">
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-cyan-400/90">
            // 02 — Key Achievements
          </p>
          <h2 className="mt-4 text-2xl font-bold text-white sm:text-3xl">What I Built</h2>
          <ul className="mt-10 grid gap-4 sm:grid-cols-1">
            {ACHIEVEMENTS.map((a, i) => (
              <li
                key={i}
                className="group flex gap-5 rounded-xl border border-white/[0.06] bg-white/[0.02] px-6 py-5 animate-fade-in-up transition hover:border-cyan-500/20 hover:bg-white/[0.04]"
                style={{ animationDelay: `${i * 80}ms`, opacity: 0 }}
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-500/10 font-mono text-sm font-semibold text-cyan-400 ring-1 ring-cyan-500/20">
                  {i + 1}
                </span>
                <span className="text-zinc-300 leading-relaxed">{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 03 — Model Results */}
        <section className="mb-28">
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-cyan-400/90">
            // 03 — Model Results
          </p>
          <h2 className="mt-4 text-2xl font-bold text-white sm:text-3xl">SMOTE vs Baseline</h2>
          <p className="mt-3 text-zinc-400">
            RandomForest performance with and without SMOTE resampling.
          </p>

          <div className="mt-12 grid gap-8 sm:grid-cols-2">
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
        <footer className="border-t border-white/[0.06] py-12">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="group inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-zinc-500 transition hover:bg-white/[0.04] hover:text-cyan-400"
          >
            <span className="transition-transform group-hover:-translate-x-0.5">←</span>
            Back to Reece Rodrigues
          </a>
        </footer>
      </main>
    </div>
  );
}

function VizCard({ title, img, alt }: { title: string; img: string; alt: string }) {
  return (
    <div className="group overflow-hidden rounded-xl border border-white/[0.08] bg-white/[0.02] shadow-lg transition-all duration-300 hover:border-cyan-500/25 hover:shadow-cyan-500/5 hover:shadow-xl">
      <div className="border-b border-white/[0.06] bg-white/[0.03] px-5 py-4">
        <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
      </div>
      <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden bg-zinc-900/80 p-5">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="max-h-full max-w-full object-contain transition-transform duration-500 group-hover:scale-105"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden items-center justify-center bg-zinc-900/95 p-6 text-center backdrop-blur-sm"
          style={{ display: "none" }}
        >
          <p className="text-sm text-zinc-400">
            Run <code className="rounded bg-white/[0.08] px-2 py-1 font-mono text-xs text-cyan-400/90">python src/Fraud_Detection.py</code> then copy <code className="rounded bg-white/[0.08] px-2 py-1 font-mono text-xs text-cyan-400/90">static/*.png</code> to <code className="rounded bg-white/[0.08] px-2 py-1 font-mono text-xs text-cyan-400/90">showcase/public/</code>
          </p>
        </div>
      </div>
    </div>
  );
}
