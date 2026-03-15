export default function Hero() {
  const scrollToMetrics = () => {
    document.getElementById('metrics')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section className="relative min-h-[60vh] flex items-center py-20">
      {/* Gradient background */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl dark:bg-blue-500/5" />
        <div className="absolute -bottom-20 -left-40 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl dark:bg-purple-500/5" />
      </div>

      <div className="mx-auto max-w-6xl px-6 w-full">
        <div className="space-y-8 max-w-3xl">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800/60">
            <span className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-pulse" />
            <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 tracking-wide">
              MACHINE LEARNING PROJECT
            </p>
          </div>

          {/* Main headline */}
          <div className="space-y-4">
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-zinc-900 dark:text-white leading-tight">
              Fraud Transaction<br />Detection System
            </h1>

            <p className="text-lg md:text-xl text-zinc-600 dark:text-zinc-400 max-w-2xl leading-relaxed">
              An advanced machine learning pipeline that detects fraudulent transactions using Random Forest classification with SMOTE for handling imbalanced datasets. Production-ready and fully containerized.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 pt-4">
            <button
              onClick={scrollToMetrics}
              className="btn-primary"
            >
              <span>View Results</span>
              <span>↓</span>
            </button>
            <a
              href="https://github.com/XreeceX/DevOps-project"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
            >
              <span>Github Repository</span>
              <span>→</span>
            </a>
          </div>

          {/* Tech badges */}
          <div className="flex flex-wrap gap-2 pt-4">
            {['Python', 'Scikit-learn', 'Docker', 'React', 'Random Forest'].map((tech) => (
              <span
                key={tech}
                className="px-3 py-1 text-xs font-medium text-zinc-700 dark:text-zinc-300 bg-zinc-100 dark:bg-white/5 rounded-full border border-zinc-200 dark:border-white/10"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
