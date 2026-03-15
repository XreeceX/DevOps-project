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
  { label: 'Model', value: 'Random Forest (100 estimators)' },
  { label: 'Data Processing', value: 'Pandas & NumPy' },
  { label: 'Visualization', value: 'Matplotlib & Seaborn' },
  { label: 'Containerization', value: 'Docker' },
  { label: 'Frontend', value: 'React + Vite + TailwindCSS' },
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
      <div className="mx-auto max-w-6xl px-6 py-16">
        {/* Hero Section */}
        <section className="mb-24 relative">
          <div className="absolute -inset-x-20 -top-20 h-96 bg-gradient-to-b from-blue-500/10 to-transparent rounded-full blur-3xl -z-10 dark:from-blue-500/5" />

          <div className="space-y-6">
            <div className="inline-block">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100 dark:bg-blue-900/30 border border-blue-300 dark:border-blue-700">
                <span className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-pulse" />
                <p className="text-sm font-semibold text-blue-600 dark:text-blue-400">Machine Learning Project</p>
              </div>
            </div>

            <h1 className="text-5xl lg:text-6xl font-bold tracking-tight text-zinc-900 dark:text-white leading-tight">
              Fraud Transaction<br />Detection System
            </h1>

            <p className="text-lg text-zinc-700 dark:text-zinc-400 max-w-2xl">
              An advanced machine learning system that detects fraudulent transactions using Random Forest classification with SMOTE for handling imbalanced datasets.
            </p>
          </div>
        </section>

        {/* Key Metrics Section */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">Model Performance Metrics</h2>
            <p className="section-subtitle">Key performance indicators from the trained models</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {METRICS.map((metric) => (
              <div key={metric.label} className="card p-6 hover:shadow-lg transition-shadow">
                <div className="text-3xl mb-3">{metric.icon}</div>
                <p className="text-sm uppercase tracking-widest font-semibold text-zinc-600 dark:text-zinc-400">
                  {metric.label}
                </p>
                <p className="text-2xl font-bold text-zinc-900 dark:text-white mt-2">
                  {metric.value}
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-500 mt-2">
                  {metric.subtitle}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Key Features Section */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">Key Features</h2>
            <p className="section-subtitle">What makes this project stand out</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {KEY_FEATURES.map((feature) => (
              <div key={feature.title} className="card p-8">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-zinc-900 dark:text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-zinc-600 dark:text-zinc-400">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Model Comparison Section */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">SMOTE Impact Analysis</h2>
            <p className="section-subtitle">Comparison of model performance with and without SMOTE data balancing</p>
          </div>

          <div className="card p-8">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-zinc-200 dark:border-zinc-700">
                    <th className="text-left py-4 px-4 font-semibold text-zinc-900 dark:text-white">Metric</th>
                    <th className="text-left py-4 px-4 font-semibold text-zinc-900 dark:text-white">Without SMOTE</th>
                    <th className="text-left py-4 px-4 font-semibold text-zinc-900 dark:text-white">With SMOTE</th>
                    <th className="text-left py-4 px-4 font-semibold text-zinc-900 dark:text-white">Improvement</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-zinc-100 dark:border-zinc-800">
                    <td className="py-4 px-4 text-zinc-700 dark:text-zinc-300">Accuracy</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">70.25%</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">68.75%</td>
                    <td className="py-4 px-4 text-red-600">-1.50%</td>
                  </tr>
                  <tr className="border-b border-zinc-100 dark:border-zinc-800">
                    <td className="py-4 px-4 text-zinc-700 dark:text-zinc-300">Precision (macro)</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">35.13%</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">41.20%</td>
                    <td className="py-4 px-4 text-green-600">+6.07%</td>
                  </tr>
                  <tr className="border-b border-zinc-100 dark:border-zinc-800">
                    <td className="py-4 px-4 text-zinc-700 dark:text-zinc-300">Recall (macro)</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">50.00%</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">49.17%</td>
                    <td className="py-4 px-4 text-red-600">-0.83%</td>
                  </tr>
                  <tr className="border-b border-zinc-100 dark:border-zinc-800">
                    <td className="py-4 px-4 text-zinc-700 dark:text-zinc-300">F1-Score (macro)</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">41.26%</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">41.50%</td>
                    <td className="py-4 px-4 text-green-600">+0.24%</td>
                  </tr>
                  <tr>
                    <td className="py-4 px-4 text-zinc-700 dark:text-zinc-300">AUC-ROC</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">0.5074</td>
                    <td className="py-4 px-4 text-zinc-900 dark:text-white font-semibold">0.5102</td>
                    <td className="py-4 px-4 text-green-600">+0.0028</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-6">
              <strong>Key Insight:</strong> SMOTE improves precision by 6.07%, helping reduce false positives when identifying fraud. The slight accuracy trade-off is acceptable for better fraud detection reliability.
            </p>
          </div>
        </section>

        {/* Visualizations Section */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">Model Visualizations</h2>
            <p className="section-subtitle">Detailed analysis charts and performance representations</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {VISUALIZATIONS.map((viz) => (
              <div key={viz.title} className="card overflow-hidden hover:shadow-lg transition-shadow">
                <div className="aspect-video bg-zinc-100 dark:bg-zinc-800 relative overflow-hidden">
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
                  <p className="text-sm text-zinc-600 dark:text-zinc-400">
                    {viz.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Technology Stack */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">Technology Stack</h2>
            <p className="section-subtitle">Tools and frameworks used in this project</p>
          </div>

          <div className="card p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {TECH_STACK.map((item) => (
                <div key={item.label} className="space-y-2">
                  <p className="text-sm uppercase tracking-widest font-semibold text-zinc-600 dark:text-zinc-400">
                    {item.label}
                  </p>
                  <p className="text-lg font-bold text-zinc-900 dark:text-white">
                    {item.value}
                  </p>
                  <div className="h-1 w-8 bg-blue-600 rounded-full" />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Project Info */}
        <section className="mb-16">
          <div className="mb-12">
            <h2 className="section-title">About This Project</h2>
          </div>

          <div className="card p-8 space-y-4">
            <div>
              <h3 className="font-bold text-zinc-900 dark:text-white mb-2">Objective</h3>
              <p className="text-zinc-700 dark:text-zinc-300">
                Build and compare machine learning models for detecting fraudulent transactions, with emphasis on handling imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique).
              </p>
            </div>
            <div>
              <h3 className="font-bold text-zinc-900 dark:text-white mb-2">Approach</h3>
              <ul className="text-zinc-700 dark:text-zinc-300 space-y-2 list-disc list-inside">
                <li>Data preprocessing: Categorical encoding and feature engineering</li>
                <li>Class imbalance handling: Comparing standard vs SMOTE-based approaches</li>
                <li>Random Forest classification with 100 estimators and class weighting</li>
                <li>Comprehensive evaluation: Accuracy, Precision, Recall, F1-Score, AUC-ROC</li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold text-zinc-900 dark:text-white mb-2">Results</h3>
              <p className="text-zinc-700 dark:text-zinc-300">
                The SMOTE-enhanced model improved precision by 6.07% in detecting fraudulent transactions, demonstrating the effectiveness of handling class imbalance in binary classification tasks.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-zinc-200 dark:border-white/10 pt-12 pb-6">
          <p className="text-sm text-zinc-700 dark:text-zinc-600">
            © 2026 Fraud Detection Project. Built with Python, Scikit-learn, React + Vite + TailwindCSS
          </p>
        </footer>
      </div>
    </main>
  )
}
