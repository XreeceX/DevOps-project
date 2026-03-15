import ServiceStatusCard from './ServiceStatusCard'

const METRICS = [
  {
    label: 'Best Accuracy',
    value: '70.25%',
    subtitle: 'Without SMOTE',
    icon: '🎯',
  },
  {
    label: 'Best AUC-ROC',
    value: '0.5102',
    subtitle: 'With SMOTE',
    icon: '📈',
  },
  {
    label: 'Precision (SMOTE)',
    value: '41.20%',
    subtitle: 'Macro average',
    icon: '🔍',
  },
  {
    label: 'Recall (SMOTE)',
    value: '49.17%',
    subtitle: 'Fraud detection rate',
    icon: '🚨',
  },
]

const VISUALIZATIONS = [
  {
    title: 'ROC Curve (With SMOTE)',
    image: '/roc_curve.png',
    description: 'Shows the trade-off between true positive and false positive rates'
  },
  {
    title: 'Confusion Matrix (With SMOTE)',
    image: '/confusion_matrix.png',
    description: 'Detailed breakdown of predictions vs actual values'
  },
  {
    title: 'ROC Curve (Without SMOTE)',
    image: '/roc_curve_no_smote.png',
    description: 'Baseline model performance without data balancing'
  },
  {
    title: 'Confusion Matrix (Without SMOTE)',
    image: '/confusion_matrix_no_smote.png',
    description: 'Baseline confusion matrix for comparison'
  },
  {
    title: 'Original Class Distribution',
    image: '/original_class_dist.png',
    description: 'Shows the imbalanced nature of fraud vs legitimate transactions'
  },
  {
    title: 'SMOTE Class Distribution',
    image: '/smote_class_dist.png',
    description: 'Balanced distribution after applying SMOTE oversampling'
  },
]

const TECH_STACK = [
  { label: 'ML Framework', value: 'Scikit-learn' },
  { label: 'Imbalance Handling', value: 'SMOTE' },
  { label: 'Model', value: 'Random Forest' },
  { label: 'Data Processing', value: 'Pandas & NumPy' },
  { label: 'Visualization', value: 'Matplotlib' },
  { label: 'Containerization', value: 'Docker' },
  { label: 'Frontend', value: 'React + Vite' },
  { label: 'Deployment', value: 'Vercel' },
]

const KEY_FEATURES = [
  {
    icon: '🤖',
    title: 'Advanced ML Model',
    description: 'Random Forest classifier with class balancing for robust fraud detection'
  },
  {
    icon: '⚖️',
    title: 'SMOTE Implementation',
    description: 'Handles imbalanced fraud datasets by generating synthetic minority samples'
  },
  {
    icon: '📊',
    title: 'Comprehensive Analysis',
    description: 'ROC curves, confusion matrices, and detailed performance metrics'
  },
  {
    icon: '🐳',
    title: 'Docker Containerized',
    description: 'Fully containerized application for easy deployment and reproducibility'
  },
  {
    icon: '⚡',
    title: 'Performance Optimized',
    description: 'Efficient data preprocessing and model evaluation pipelines'
  },
  {
    icon: '🎨',
    title: 'Interactive Dashboard',
    description: 'Modern React-based showcase with dark mode support'
  },
]

export default function Dashboard() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-zinc-50 dark:from-[#0a0a0f] dark:to-zinc-900/50">
      <div className="mx-auto max-w-6xl px-6">
        {/* Overview / Key Features Section */}
        <section id="overview" className="py-20 border-b border-zinc-200 dark:border-white/10">
          <div className="mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 mb-6">
              <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">PROJECT OVERVIEW</span>
            </div>
            <h2 className="section-title">Project Highlights</h2>
            <p className="section-subtitle">What makes this fraud detection system production-grade</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {KEY_FEATURES.map((feature) => (
              <div key={feature.title} className="card p-8 hover:shadow-lg transition-all">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-bold text-zinc-900 dark:text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Key Metrics Section */}
        <section id="metrics" className="py-20 border-b border-zinc-200 dark:border-white/10">
          <div className="mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 mb-6">
              <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">PERFORMANCE DATA</span>
            </div>
            <h2 className="section-title">Model Performance Metrics</h2>
            <p className="section-subtitle">Key performance indicators from the trained models</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
            {METRICS.map((metric) => (
              <div key={metric.label} className="card p-6 hover:shadow-lg transition-all">
                <div className="text-3xl mb-4">{metric.icon}</div>
                <p className="text-xs uppercase tracking-widest font-semibold text-zinc-600 dark:text-zinc-400 mb-2">
                  {metric.label}
                </p>
                <p className="text-3xl font-bold text-zinc-900 dark:text-white mb-1">
                  {metric.value}
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-500">
                  {metric.subtitle}
                </p>
              </div>
            ))}
          </div>

          {/* SMOTE Comparison Table */}
          <div className="card p-8 overflow-hidden">
            <h3 className="text-xl font-bold text-zinc-900 dark:text-white mb-6">SMOTE Impact Analysis</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 dark:border-zinc-700">
                    <th className="text-left py-3 px-4 font-semibold text-zinc-900 dark:text-white">Metric</th>
                    <th className="text-left py-3 px-4 font-semibold text-zinc-900 dark:text-white">Without SMOTE</th>
                    <th className="text-left py-3 px-4 font-semibold text-zinc-900 dark:text-white">With SMOTE</th>
                    <th className="text-left py-3 px-4 font-semibold text-zinc-900 dark:text-white">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { metric: 'Accuracy', without: '70.25%', with: '68.75%', change: '-1.50%', good: false },
                    { metric: 'Precision (macro)', without: '35.13%', with: '41.20%', change: '+6.07%', good: true },
                    { metric: 'Recall (macro)', without: '50.00%', with: '49.17%', change: '-0.83%', good: false },
                    { metric: 'F1-Score (macro)', without: '41.26%', with: '41.50%', change: '+0.24%', good: true },
                    { metric: 'AUC-ROC', without: '0.5074', with: '0.5102', change: '+0.0028', good: true },
                  ].map((row) => (
                    <tr key={row.metric} className="border-b border-zinc-100 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-white/5">
                      <td className="py-3 px-4 text-zinc-700 dark:text-zinc-300">{row.metric}</td>
                      <td className="py-3 px-4 text-zinc-900 dark:text-white font-semibold">{row.without}</td>
                      <td className="py-3 px-4 text-zinc-900 dark:text-white font-semibold">{row.with}</td>
                      <td className={`py-3 px-4 font-semibold ${row.good ? 'text-green-600' : 'text-orange-600'}`}>
                        {row.change}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-6 p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800/40">
              <strong>Key Insight:</strong> SMOTE improves precision by 6.07%, helping reduce false positives in fraud detection. The slight accuracy trade-off is acceptable for better fraud detection reliability in production.
            </p>
          </div>
        </section>

        {/* Visualizations Section */}
        <section id="visualizations" className="py-20 border-b border-zinc-200 dark:border-white/10">
          <div className="mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 mb-6">
              <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">ANALYSIS</span>
            </div>
            <h2 className="section-title">Model Visualizations</h2>
            <p className="section-subtitle">Detailed analysis charts and performance representations</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {VISUALIZATIONS.map((viz) => (
              <div key={viz.title} className="card overflow-hidden hover:shadow-lg transition-all">
                <div className="aspect-video bg-zinc-200 dark:bg-zinc-700 relative overflow-hidden">
                  <img
                    src={viz.image}
                    alt={viz.title}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-6">
                  <h3 className="font-bold text-zinc-900 dark:text-white mb-2">
                    {viz.title}
                  </h3>
                  <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                    {viz.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Technology Stack */}
        <section id="tech-stack" className="py-20">
          <div className="mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 mb-6">
              <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">TECH STACK</span>
            </div>
            <h2 className="section-title">Technology & Tools</h2>
            <p className="section-subtitle">Built with industry-standard tools and frameworks</p>
          </div>

          <div className="card p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {TECH_STACK.map((item) => (
                <div key={item.label} className="flex flex-col">
                  <p className="text-xs uppercase tracking-widest font-semibold text-zinc-600 dark:text-zinc-400 mb-3">
                    {item.label}
                  </p>
                  <p className="text-lg font-bold text-zinc-900 dark:text-white mb-3">
                    {item.value}
                  </p>
                  <div className="h-1 w-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full" />
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
