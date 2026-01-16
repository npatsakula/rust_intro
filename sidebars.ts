import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  courseSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Block I: From C++ to Rust',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Lecture 1: Ecosystem & Ownership',
          items: [
            'lectures/lecture-01',
            'seminars/seminar-01-1',
            'seminars/seminar-01-2',
          ],
        },
        {
          type: 'category',
          label: 'Lecture 2: Types, Traits & Modules',
          items: [
            'lectures/lecture-02',
            'seminars/seminar-02-1',
            'seminars/seminar-02-2',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Block II: Mathematical Tooling',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Lecture 3: Linear Algebra & Tensors',
          items: [
            'lectures/lecture-03',
            'seminars/seminar-03-1',
            'seminars/seminar-03-2',
          ],
        },
        {
          type: 'category',
          label: 'Lecture 4: Statistics & ML',
          items: [
            'lectures/lecture-04',
            'seminars/seminar-04-1',
            'seminars/seminar-04-2',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Block III: Domain Applications',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Lecture 5: Numerical Methods & FEM',
          items: [
            'lectures/lecture-05',
            'seminars/seminar-05-1',
            'seminars/seminar-05-2',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Block IV: Advanced Rust',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Lecture 6: Unsafe & FFI',
          items: [
            'lectures/lecture-06',
            'seminars/seminar-06-1',
            'seminars/seminar-06-2',
          ],
        },
        {
          type: 'category',
          label: 'Lecture 7: Verification & Production',
          items: [
            'lectures/lecture-07',
            'seminars/seminar-07-1',
            'seminars/seminar-07-2',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      collapsed: true,
      items: [
        'appendix/resources',
        'appendix/assessment',
        'appendix/final-project',
      ],
    },
  ],
};

export default sidebars;
