/** @type {import('next').NextConfig} */
const nextConfig = {
  // remove `output: 'export'` so Vercel can use its default build output
  // which produces the required routes-manifest.json in .next.
  images: { unoptimized: true },
};

module.exports = nextConfig;
