import { videoPanel } from './video.js';
import { servosPanel } from './servos.js';
import { ikPanel } from './ik.js';
import { movePanel } from './move.js';
import { terminalPanel } from './terminal.js';
import { logsPanel } from './logs.js';
import { pythonPanel } from './python.js';
import { systemPanel } from './system.js';
import { datacollectPanel } from './datacollect.js';
import { calibrationPanel } from './calibration.js';
import { filesPanel } from './files.js';

export const PANELS = [
  videoPanel,
  servosPanel,
  ikPanel,
  movePanel,
  terminalPanel,
  logsPanel,
  pythonPanel,
  systemPanel,
  calibrationPanel,
  datacollectPanel,
  filesPanel,
];
