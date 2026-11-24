import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Controls } from './components/Controls'
import { ChangeLog, Contributions, Readout, type LogEntry } from './components/Readout'
import { Viewport, type Change } from './components/Viewport'
import type { Body, Config, ModelBundle, ModelName } from './model'
import { defaultConfig, fmt, predict } from './model'
import type { Part } from './scene/Studio'
import { BODIES } from './scene/Studio'

const REPO = 'https://github.com/WonDeBuk/CSC14005-Lab01-Linear-Regression'
const PART_OF: Record<keyof Config, Part> = {
  BODY: 'body',
  ENGINESIZE: 'engine',
  CYLINDERS: 'engine',
  GEARS: 'gearbox',
  TRANS_TYPE: 'gearbox',
  HYBRID: 'hybrid',
  FUELTYPE: 'fuel',
}
const PIXEL_SIZES = [1, 3, 4, 6]

export default function App() {
  const [bundle, setBundle] = useState<ModelBundle | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}model.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setBundle)
      .catch(() => setFailed(true))
  }, [])

  if (failed) return <p className="boot">Không tải được mô hình. Tải lại trang để thử lại.</p>
  if (!bundle) return <p className="boot">Đang tải mô hình…</p>
  return <Workshop bundle={bundle} />
}

function describe(bundle: ModelBundle, key: keyof Config, from: Config, to: Config): string {
  const L = bundle.labels
  switch (key) {
    case 'BODY':
      return `Thân xe: ${L.BODY[from.BODY]} → ${L.BODY[to.BODY]}`
    case 'FUELTYPE':
      return `Nhiên liệu: ${L.FUELTYPE[from.FUELTYPE]} → ${L.FUELTYPE[to.FUELTYPE]}`
    case 'TRANS_TYPE':
      return `Hộp số: ${from.TRANS_TYPE} → ${to.TRANS_TYPE}`
    case 'ENGINESIZE':
      return `Dung tích: ${fmt(from.ENGINESIZE, 1)} → ${fmt(to.ENGINESIZE, 1)} L`
    case 'CYLINDERS':
      return `Xi-lanh: ${from.CYLINDERS} → ${to.CYLINDERS}`
    case 'GEARS':
      return `Số cấp: ${from.GEARS || 'CVT'} → ${to.GEARS || 'CVT'}`
    case 'HYBRID':
      return to.HYBRID ? 'Bật hybrid' : 'Tắt hybrid'
  }
}

function Workshop({ bundle }: { bundle: ModelBundle }) {
  const [modelName, setModelName] = useState<ModelName>('M8b')
  const [config, setConfig] = useState<Config>(() => defaultConfig(bundle))
  const [mode, setMode] = useState<'single' | 'lineup'>('single')
  const [pixelSize, setPixelSize] = useState(4)
  const [xray, setXray] = useState(false)
  const [change, setChange] = useState<Change | null>(null)
  const [log, setLog] = useState<LogEntry[]>([])
  const counter = useRef(0)

  const model = bundle.models[modelName]
  const prediction = useMemo(() => predict(model, bundle.fleet, config), [model, bundle.fleet, config])
  const lineup = useMemo(() => {
    const configs = BODIES.map((b) => ({ ...config, BODY: b }))
    return { configs, co2: configs.map((c) => predict(model, bundle.fleet, c).mean) }
  }, [model, bundle.fleet, config])

  const apply = useCallback(
    (next: Config, key: keyof Config) => {
      const before = predict(model, bundle.fleet, config).mean
      const after = predict(model, bundle.fleet, next).mean
      const id = ++counter.current
      setConfig(next)
      setChange({ id, part: PART_OF[key], delta: after - before })
      setLog((l) => [{ id, text: describe(bundle, key, config, next), delta: after - before }, ...l].slice(0, 6))
    },
    [bundle, config, model],
  )

  const onChange = useCallback(
    <K extends keyof Config>(key: K, value: Config[K]) => {
      if (config[key] === value) return
      const next = { ...config, [key]: value }
      // A CVT has no fixed gear count in the data; leaving CVT restores a typical 6 gears.
      if (key === 'TRANS_TYPE') next.GEARS = value === 'AV' ? 0 : config.GEARS === 0 ? 6 : config.GEARS
      apply(next, key)
    },
    [apply, config],
  )

  const onReset = useCallback(() => {
    apply({ ...config, ...bundle.presets[config.BODY] }, 'ENGINESIZE')
  }, [apply, bundle.presets, config])

  const onSelectBody = useCallback((body: Body) => onChange('BODY', body), [onChange])

  return (
    <div className="app">
      <a className="skip-link" href="#controls-title">
        Bỏ qua tới bảng thông số
      </a>
      <header className="topbar">
        <a className="wordmark" href="./" aria-label="CO₂ Studio, trang chủ">
          <span className="wordmark-mark" aria-hidden="true">
            CO₂
          </span>
          Studio
        </a>
        <p className="topbar-meta mono">Lab 01 · Hồi quy tuyến tính · 1.067 xe Canada 2014</p>
        <nav className="topbar-nav" aria-label="Liên kết">
          <a className="link" href={REPO}>
            Mã nguồn
          </a>
          <a className="link" href={`${REPO}/blob/main/data/README.md`}>
            Dữ liệu
          </a>
        </nav>
      </header>

      <main className="workspace">
        <div className="stage">
          <h1 className="stage-title">Ước tính CO₂ của xe từ thông số thiết kế</h1>
          <div className="toolbar" role="toolbar" aria-label="Chế độ xem">
            <div className="chip-group" role="group" aria-label="Chế độ">
              <button type="button" className="chip" aria-pressed={mode === 'single'} onClick={() => setMode('single')}>
                Một xe
              </button>
              <button type="button" className="chip" aria-pressed={mode === 'lineup'} onClick={() => setMode('lineup')}>
                Đội hình 5 xe
              </button>
            </div>
            <div className="chip-group" role="group" aria-label="Cỡ điểm ảnh">
              {PIXEL_SIZES.map((p) => (
                <button key={p} type="button" className="chip" aria-pressed={pixelSize === p} onClick={() => setPixelSize(p)}>
                  {p === 1 ? 'Tắt pixel' : `${p} px`}
                </button>
              ))}
            </div>
            <button type="button" className="chip" aria-pressed={xray} onClick={() => setXray((x) => !x)}>
              Nhìn xuyên thân xe
            </button>
          </div>

          <Viewport
            config={config}
            prediction={prediction}
            fleet={bundle.fleet}
            change={change}
            mode={mode}
            pixelSize={pixelSize}
            xray={xray}
            lineup={lineup}
            onSelectBody={onSelectBody}
          />

          <div className="model-strip">
            <div className="chip-group" role="group" aria-label="Mô hình">
              {(Object.keys(bundle.models) as ModelName[]).map((name) => (
                <button key={name} type="button" className="chip" aria-pressed={name === modelName} onClick={() => setModelName(name)}>
                  {name}
                </button>
              ))}
            </div>
            <p className="model-desc">{model.description}</p>
            <dl className="model-metrics mono">
              <div>
                <dt>R² test</dt>
                <dd>{fmt(model.metrics.R2, 3)}</dd>
              </div>
              <div>
                <dt>RMSE test</dt>
                <dd>{fmt(model.metrics.RMSE, 1)}</dd>
              </div>
              <div>
                <dt>RMSE CV 10-fold</dt>
                <dd>
                  {fmt(model.metrics.CV_RMSE_mean, 1)} ± {fmt(model.metrics.CV_RMSE_std, 1)}
                </dd>
              </div>
            </dl>
          </div>
        </div>

        <aside className="panel" aria-label="Bảng điều khiển">
          <Readout prediction={prediction} model={model} />
          <Controls config={config} bundle={bundle} onChange={onChange} onReset={onReset} />
          <Contributions prediction={prediction} />
          <ChangeLog entries={log} />
        </aside>
      </main>

      <footer className="footer">
        <p>
          Mẫu xe: Kenney Car Kit (CC0) · Dữ liệu: Natural Resources Canada, Fuel Consumption Ratings 2014 · Nhóm CSC14005,
          Khoa CNTT, ĐH Khoa học Tự nhiên, ĐHQG-HCM
        </p>
      </footer>
    </div>
  )
}
