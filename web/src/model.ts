// Linear model evaluated in the browser from public/model.json
// (written by `python -m linreg.export_web`).

export type Body = 'small' | 'midlarge' | 'suv' | 'pickup' | 'van'
export type Fuel = 'X' | 'Z' | 'D' | 'E'
export type Trans = 'A' | 'AS' | 'AM' | 'AV' | 'M'
export type ModelName = 'M3h' | 'M8b'

export interface Config {
  BODY: Body
  FUELTYPE: Fuel
  TRANS_TYPE: Trans
  ENGINESIZE: number
  CYLINDERS: number
  GEARS: number
  HYBRID: 0 | 1
}

export interface ExportedModel {
  name: ModelName
  description: string
  features: string[]
  coef: number[]
  xtx_inv: number[][]
  sigma2: number
  t_crit: number
  leverage_limit: number
  metrics: Record<'MAE' | 'RMSE' | 'R2' | 'CV_RMSE_mean' | 'CV_RMSE_std', number>
}

export interface ModelBundle {
  source: string
  models: Record<ModelName, ExportedModel>
  fleet: number[]
  ranges: Record<'ENGINESIZE' | 'CYLINDERS' | 'GEARS', [number, number]>
  labels: { BODY: Record<Body, string>; FUELTYPE: Record<Fuel, string>; TRANS_TYPE: Record<Trans, string> }
  presets: Record<Body, { ENGINESIZE: number; CYLINDERS: number; GEARS: number }>
}

export type Factor = 'engine' | 'cylinders' | 'transmission' | 'fuel' | 'body' | 'hybrid'

export interface Prediction {
  mean: number
  ci: [number, number]
  pi: [number, number]
  leverage: number
  extrapolating: boolean
  /** Share of the 2014 fleet that emits less than this configuration. */
  percentile: number
  contributions: { factor: Factor; value: number }[]
  intercept: number
}

export const FACTOR_LABELS: Record<Factor, string> = {
  engine: 'Dung tích động cơ',
  cylinders: 'Số xi-lanh',
  transmission: 'Hộp số',
  fuel: 'Nhiên liệu',
  body: 'Nhóm thân xe',
  hybrid: 'Hybrid',
}

const FACTOR_OF: Record<string, Factor> = {
  ENGINESIZE: 'engine',
  CYLINDERS: 'cylinders',
  GEARS: 'transmission',
  NOGEAR: 'transmission',
  HYBRID: 'hybrid',
  ENGINE_X_HYBRID: 'hybrid',
  ENGINE_X_PICKUP: 'body',
  ENGINE_X_VAN: 'body',
}

function featureValue(name: string, c: Config): number {
  switch (name) {
    case '(intercept)':
      return 1
    case 'ENGINESIZE':
      return c.ENGINESIZE
    case 'CYLINDERS':
      return c.CYLINDERS
    case 'GEARS':
      return c.GEARS
    case 'NOGEAR':
      return c.GEARS === 0 ? 1 : 0
    case 'HYBRID':
      return c.HYBRID
    case 'ENGINE_X_PICKUP':
      return c.BODY === 'pickup' ? c.ENGINESIZE : 0
    case 'ENGINE_X_VAN':
      return c.BODY === 'van' ? c.ENGINESIZE : 0
    case 'ENGINE_X_HYBRID':
      return c.HYBRID * c.ENGINESIZE
  }
  // One-hot columns: FUELTYPE_Z, BODY_suv, TRANS_TYPE_AV ...
  for (const key of ['FUELTYPE', 'BODY', 'TRANS_TYPE'] as const) {
    if (name.startsWith(key + '_')) return c[key] === name.slice(key.length + 1) ? 1 : 0
  }
  throw new Error(`Unknown feature ${name}`)
}

function factorOf(name: string): Factor {
  if (FACTOR_OF[name]) return FACTOR_OF[name]
  if (name.startsWith('FUELTYPE_')) return 'fuel'
  if (name.startsWith('BODY_')) return 'body'
  return 'transmission'
}

export function predict(model: ExportedModel, fleet: number[], c: Config): Prediction {
  const x = model.features.map((f) => featureValue(f, c))
  const mean = x.reduce((s, v, i) => s + v * model.coef[i], 0)
  // Leverage h = x'(X'X)^-1 x
  const leverage = x.reduce((s, xi, i) => s + xi * x.reduce((t, xj, j) => t + model.xtx_inv[i][j] * xj, 0), 0)
  const seMean = Math.sqrt(model.sigma2 * leverage)
  const sePred = Math.sqrt(model.sigma2 * (1 + leverage))
  const t = model.t_crit

  const totals = new Map<Factor, number>()
  model.features.forEach((f, i) => {
    if (f === '(intercept)') return
    const factor = factorOf(f)
    totals.set(factor, (totals.get(factor) ?? 0) + x[i] * model.coef[i])
  })

  return {
    mean,
    ci: [mean - t * seMean, mean + t * seMean],
    pi: [mean - t * sePred, mean + t * sePred],
    leverage,
    extrapolating: leverage > model.leverage_limit,
    percentile: fleet.filter((v) => v < mean).length / fleet.length,
    contributions: [...totals].map(([factor, value]) => ({ factor, value })),
    intercept: model.coef[0],
  }
}

export function defaultConfig(bundle: ModelBundle): Config {
  return { BODY: 'small', FUELTYPE: 'X', TRANS_TYPE: 'AS', HYBRID: 0, ...bundle.presets.small }
}

export const fmt = (v: number, digits = 0) =>
  v.toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const signed = (v: number, digits = 1) => (v > 0 ? '+' : v < 0 ? '−' : '±') + fmt(Math.abs(v), digits)
