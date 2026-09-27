// Declarative workspace presets. Leaves are panel ids; a `stack` node becomes
// a tab group, `row`/`column` become splits.

const stack = (size, ...ids) => ({ dir: 'stack', size: size || undefined, children: ids.map((id) => ({ id })) });
const row = (...children) => ({ dir: 'row', children });
const col = (...children) => ({ dir: 'column', children });

export const PRESETS = {
  control: {
    label: 'Пульт',
    root: col(
      row(
        stack('25%', 'servos', 'ik'),
        stack('47%', 'video'),
        stack('28%', 'terminal', 'logs', 'python'),
      ),
      row(
        stack('76%', 'files'),
        stack('24%', 'system', 'datacollect'),
      ),
    ),
  },
  debug: {
    label: 'Отладка',
    root: col(
      stack('58%', 'terminal', 'logs', 'python'),
      stack('42%', 'system', 'datacollect'),
    ),
  },
  files: {
    label: 'Файлы',
    root: row(
      stack('26%', 'servos', 'ik', 'system'),
      stack('74%', 'files'),
    ),
  },
  collect: {
    label: 'Сбор',
    root: row(
      stack('56%', 'video'),
      stack('44%', 'datacollect', 'servos'),
    ),
  },
  mobile: {
    label: 'Мобильный',
    root: stack(
      null,
      'video', 'servos', 'ik', 'terminal', 'logs',
      'files', 'datacollect', 'python', 'system',
    ),
  },
};
