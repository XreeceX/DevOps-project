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
    <div className="relative min-h-screen mesh-bg">
      {/* Nav - portfolio-style with RR logo */}
      <nav className="sticky top-0 z-50 px-4 pt-4">
        <div className="glass-nav mx-auto flex h-14 max-w-[1200px] items-center justify-between rounded-xl px-4 shadow-card md:px-6">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="flex items-center gap-3 text-slate-200 transition hover:text-white"
          >
            <span className="font-display text-lg font-semibold tracking-tight">RR</span>
            <span className="hidden text-sm font-medium sm:inline">Reece Rodrigues</span>
          </a>
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="group flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/5 hover:text-white"
          >
            <span className="transition-transform group-hover:-translate-x-0.5">←</span>
            Back to Portfolio
          </a>
        </div>
      </nav>

      <main className="relative z-10 mx-auto max-w-[1200px] px-4 py-20 sm:px-6 sm:py-28 lg:px-8">
        {/* // 01 — Overview - portfolio project card style */}
        <section className="group relative mb-24 overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-card transition duration-300 hover:border-accent/30 sm:p-12">
          <span className="section-header-accent ml-0" aria-hidden />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-accent md:text-sm">
            // 01 — Overview
          </p>
          <h1 className="mt-4 font-display text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
            Fraud Detection
          </h1>
          <p className="mt-2 text-sm font-medium uppercase tracking-widest text-muted">
            Case Study 5 · DevOps Project
          </p>
          <p className="mt-6 max-w-2xl text-base leading-relaxed text-mutedLight md:text-lg">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class imbalance
            and RandomForest for robust predictions—containerized for reproducible deployment.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-300"
              >
                {t}
              </span>
            ))}
          </div>
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="group/btn mt-10 inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-white shadow-glow transition duration-200 hover:brightness-110"
          >
            View Code
            <span className="transition-transform group-hover/btn:translate-x-0.5">→</span>
          </a>
        </section>

        {/* // 02 — Key Achievements */}
        <section className="mb-24">
          <span className="section-header-accent ml-0 mr-auto" aria-hidden />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-accent md:text-sm">
            // 02 — Key Achievements
          </p>
          <h2 className="mt-2 font-display text-2xl font-bold text-white sm:text-3xl">What I Built</h2>
          <ul className="mt-10 space-y-3">
            {ACHIEVEMENTS.map((a, i) => (
              <li
                key={i}
                className="flex gap-4 rounded-xl border border-white/[0.08] bg-white/[0.03] px-6 py-4 transition hover:border-accent/30 hover:bg-white/[0.05]"
              >
                <span className="text-accent">▹</span>
                <span className="text-slate-300 leading-relaxed">{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 03 — Model Results */}
        <section className="mb-24">
          <span className="section-header-accent ml-0 mr-auto" aria-hidden />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-accent md:text-sm">
            // 03 — Model Results
          </p>
          <h2 className="mt-2 font-display text-2xl font-bold text-white sm:text-3xl">SMOTE vs Baseline</h2>
          <p className="mt-3 text-slate-300">
            RandomForest performance with and without SMOTE resampling.
          </p>

          <div className="mt-12 grid gap-6 sm:grid-cols-2">
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

          <div className="mt-6 grid gap-6 sm:grid-cols-2">
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

          <div className="mt-6 grid gap-6 sm:grid-cols-2">
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

        {/* Footer - portfolio style */}
        <footer className="border-t border-white/[0.08] py-12">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
            <a
              href="https://reece-rodrigues.vercel.app/"
              className="group inline-flex items-center gap-2 text-sm font-medium text-slate-400 transition hover:text-white"
            >
              <span className="transition-transform group-hover:-translate-x-0.5">←</span>
              Back to Reece Rodrigues
            </a>
            <p className="text-sm text-slate-500">
              © 2026 Reece Rodrigues. Crafted with intent.
            </p>
          </div>
          <p className="mt-4 text-xs text-slate-600">
            London, UK · AI Engineer & Full-Stack Developer
          </p>
        </footer>
      </main>
    </div>
  );
}

function VizCard({ title, img, alt }: { title: string; img: string; alt: string }) {
  return (
    <div className="group overflow-hidden rounded-xl border border-white/[0.08] bg-white/[0.03] shadow-card transition duration-200 hover:border-white/15 hover:bg-white/[0.05]">
      <div className="border-b border-white/[0.06] px-5 py-4">
        <h3 className="font-display text-base font-semibold text-white">{title}</h3>
      </div>
      <div className="relative flex min-h-[220px] aspect-[4/3] items-center justify-center overflow-hidden bg-panel/80 p-6">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="max-h-full max-w-full object-contain transition duration-300 group-hover:scale-[1.02]"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden flex-col items-center justify-center gap-3 bg-panel/95 p-6 text-center"
          style={{ display: "none" }}
        >
          <p className="text-sm font-medium text-slate-400">Charts not generated yet</p>
          <p className="max-w-sm text-xs leading-relaxed text-slate-500">
            Run <code className="rounded bg-white/5 px-2 py-1 font-mono text-accentSoft">python src/Fraud_Detection.py</code> then copy <code className="rounded bg-white/5 px-2 py-1 font-mono text-accentSoft">static/*.png</code> to <code className="rounded bg-white/5 px-2 py-1 font-mono text-accentSoft">showcase/public/</code>
          </p>
        </div>
      </div>
    </div>
  );
}
