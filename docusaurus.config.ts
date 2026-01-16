import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Rust for Scientific Computing',
  tagline: 'A university course for applied mathematicians',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://your-university.example.com',
  baseUrl: '/rust-course/',

  organizationName: 'your-university',
  projectName: 'rust-course',

  onBrokenLinks: 'throw',

  // Internationalization configuration
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ru'],
    localeConfigs: {
      en: {
        htmlLang: 'en-US',
        label: 'English',
      },
      ru: {
        htmlLang: 'ru-RU',
        label: 'Русский',
      },
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/', // Serve docs at the site's root
          editUrl: 'https://github.com/your-university/rust-course/edit/main/website/',
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/rust-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    navbar: {
      title: 'Rust Course',
      logo: {
        alt: 'Rust Logo',
        src: 'img/rust-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'courseSidebar',
          position: 'left',
          label: 'Course',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/your-university/rust-course',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Course',
          items: [
            {
              label: 'Introduction',
              to: '/',
            },
            {
              label: 'Lectures',
              to: '/lectures/lecture-01',
            },
            {
              label: 'Seminars',
              to: '/seminars/seminar-01-1',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Rust Book',
              href: 'https://doc.rust-lang.org/book/',
            },
            {
              label: 'Rustonomicon',
              href: 'https://doc.rust-lang.org/nomicon/',
            },
            {
              label: 'Rust Playground',
              href: 'https://play.rust-lang.org/',
            },
          ],
        },
        {
          title: 'Scientific Libraries',
          items: [
            {
              label: 'nalgebra',
              href: 'https://nalgebra.org/',
            },
            {
              label: 'russell',
              href: 'https://github.com/cpmech/russell',
            },
            {
              label: 'linfa',
              href: 'https://rust-ml.github.io/linfa/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Faculty of Applied Math and Computer Science. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml', 'bash', 'cpp', 'c'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
