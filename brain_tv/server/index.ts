import express from 'express';
import cors from 'cors';
import { runsRouter } from './routes/runs.js';
import { tensorboardRouter } from './routes/tensorboard.js';
import { actionsRouter } from './routes/actions.js';

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/runs', runsRouter);
app.use('/api/tensorboard', tensorboardRouter);
app.use('/api/actions', actionsRouter);

const PORT = 5199;
app.listen(PORT, () => {
  console.log(`Brain Factory Watch API running on http://localhost:${PORT}`);
});
