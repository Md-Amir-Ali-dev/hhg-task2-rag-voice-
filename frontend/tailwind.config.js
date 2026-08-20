/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#080b14', // Very dark blue/slate
        surface: 'rgba(255, 255, 255, 0.03)',
        border: 'rgba(255, 255, 255, 0.08)',
        success: {
          DEFAULT: '#10b981', // emerald-500
          light: '#34d399',
        },
        brand: {
          purple: '#0ea5e9', // Changed to sky-500 to match coastal vibe
          amber: '#fbbf24', // Amber/Yellow for the sun
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
