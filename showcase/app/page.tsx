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
      {/* Nav - portfolio-style glass */}
      <nav className="sticky top-0 z-50 px-4 pt-4">
        <div className="glass-nav mx-auto flex h-14 max-w-[1200px] items-center justify-between rounded-2xl border border-white/10 px-4 shadow-card md:px-6">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="group flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/5 hover:text-white"
          >
            <span className="transition-transform group-hover:-translate-x-0.5">←</span>
            Back to Portfolio
          </a>
        </div>
      </nav>

      <main className="relative z-10 mx-auto max-w-[1200px] px-4 py-16 sm:px-6 sm:py-24 lg:px-8">
        {/* // 01 — Overview - portfolio project card style */}
        <section className="group relative mb-16 overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] p-8 shadow-card transition duration-200 hover:-translate-y-0.5 hover:border-accent/50 sm:p-10">
          <span className="section-header-accent ml-0" aria-hidden />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-accent md:text-sm">
            // 01 — Overview
          </p>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-white sm:text-5xl">
            Fraud Detection
          </h1>
          <p className="mt-2 text-sm font-medium uppercase tracking-widest text-muted">
            Case Study 5 · DevOps Project
          </p>
          <p className="mt-6 max-w-2xl text-base leading-relaxed text-muted md:text-lg">
            ML-powered fraud detection for banking transactions. Uses SMOTE to address class imbalance
            and RandomForest for robust predictions—containerized for reproducible deployment.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {TECH.map((t) => (
              <span
                key={t}
                className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-slate-300"
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
        <section className="mb-16">
          <span className="section-header-accent ml-0 mr-auto" aria-hidden />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-accent md:text-sm">
            // 02 — Key Achievements
          </p>
          <h2 className="mt-2 text-2xl font-bold text-white sm:text-3xl">What I Built</h2>
          <ul className="mt-8 space-y-4">
            {ACHIEVEMENTS.map((a, i) => (
              <li
                key={i}
                className="flex gap-4 rounded-2xl border border-white/10 bg-white/[0.02] px-6 py-5 transition hover:border-accent/40 hover:bg-white/[0.04]"
              >
                <span className="text-accent">▹</span>
                <span className="text-muted leading-relaxed">{a}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* // 03 — Model Results */}
        <section className="mb-16">
          <span className="section-header-accent ml-0 mr-auto" aria-hidden />
          <p className="font-mono text-xs font-medium uppercase tracking-[0.2em] text-accent md:text-sm">
            // 03 — Model Results
          </p>
          <h2 className="mt-2 text-2xl font-bold text-white sm:text-3xl">SMOTE vs Baseline</h2>
          <p className="mt-3 text-muted">
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
        <footer className="border-t border-white/10 py-10">
          <a
            href="https://reece-rodrigues.vercel.app/"
            className="group inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-muted transition hover:bg-white/5 hover:text-accentSoft"
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
    <div className="group overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] shadow-card transition duration-200 hover:-translate-y-1 hover:border-accent/50">
      <div className="border-b border-white/10 bg-white/[0.03] px-5 py-4">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
      </div>
      <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden bg-panel/80 p-5">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={img}
          alt={alt}
          className="max-h-full max-w-full object-contain transition duration-300 group-hover:scale-[1.03]"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const placeholder = e.currentTarget.nextElementSibling;
            if (placeholder) (placeholder as HTMLElement).style.display = "flex";
          }}
        />
        <div
          className="absolute inset-0 hidden items-center justify-center bg-panel/95 p-6 text-center backdrop-blur-sm"
          style={{ display: "none" }}
        >
          <p className="text-sm text-muted">
            Run <code className="rounded bg-white/10 px-2 py-1 font-mono text-xs text-accentSoft">python src/Fraud_Detection.py</code> then copy <code className="rounded bg-white/10 px-2 py-1 font-mono text-xs text-accentSoft">static/*.png</code> to <code className="rounded bg-white/10 px-2 py-1 font-mono text-xs text-accentSoft">showcase/public/</code>
          </p>
        </div>
      </div>
    </div>
  );
}
