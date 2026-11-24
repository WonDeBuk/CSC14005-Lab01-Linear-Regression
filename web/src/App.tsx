import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Controls } from './components/Controls'
import { LogCard, ResultCard, WhyCard, type LogEntry } from './components/Readout'
import { Viewport, type Change, type ViewRequest } from './components/Viewport'
import type { Body, Config, ModelBundle, ModelName } from './model'
import { defaultConfig, fmt, predict } from './model'
import type { Part, View } from './scene/Studio'
import { BODIES, smokeColor } from './scene/Studio'

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
const PIXEL_SIZES = [1, 3, 4, 6, 8]
const VIEWS: { id: View; label: string }[] = [
  { id: 'three-quarter', label: '3/4' },
  { id: 'side', label: 'Hông' },
  { id: 'top', label: 'Trên' },
  { id: 'engine', label: 'Khoang máy' },
  { id: 'rear', label: 'Đuôi + khí xả' },
  { id: 'under', label: 'Gầm' },
]
type Tab = 'studio' | 'xray' | 'lineup'

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
      return `Số cấp: ${from.GEARS || 'không ghi'} → ${to.GEARS || 'không ghi'}`
    case 'HYBRID':
      return to.HYBRID ? 'Bật hybrid' : 'Tắt hybrid'
  }
}

function Workshop({ bundle }: { bundle: ModelBundle }) {
  const [modelName, setModelName] = useState<ModelName>('M8b')
  const [config, setConfig] = useState<Config>(() => defaultConfig(bundle))
  const [tab, setTab] = useState<Tab>('studio')
  const [pixelSize, setPixelSize] = useState(4)
  const [follow, setFollow] = useState(true)
  const [autoRotate, setAutoRotate] = useState(false)
  const [view, setView] = useState<ViewRequest | null>(null)
  const [thumbs, setThumbs] = useState<Partial<Record<Body, string>>>({})
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
    (next: Config, part: Part, text: string) => {
      const delta = predict(model, bundle.fleet, next).mean - predict(model, bundle.fleet, config).mean
      const id = ++counter.current
      setConfig(next)
      setChange({ id, part, delta })
      setLog((l) => [{ id, text, delta }, ...l].slice(0, 7))
    },
    [bundle.fleet, config, model],
  )

  const onChange = useCallback(
    <K extends keyof Config>(key: K, value: Config[K]) => {
      if (config[key] === value) return
      const next = { ...config, [key]: value }
      // A CVT has no fixed gear count in the data; leaving CVT restores a typical 6 gears.
      if (key === 'TRANS_TYPE') next.GEARS = value === 'AV' ? 0 : config.GEARS === 0 ? 6 : config.GEARS
      apply(next, PART_OF[key], describe(bundle, key, config, next))
    },
    [apply, bundle, config],
  )

  const onPreset = useCallback((preset: Config) => apply(preset, 'body', `Thử nhanh: ${bundle.labels.BODY[preset.BODY]}, ${fmt(preset.ENGINESIZE, 1)} L`), [apply, bundle])
  const onSelectBody = useCallback((body: Body) => onChange('BODY', body), [onChange])
  const requestView = (v: View) => setView({ id: ++counter.current, view: v })

  const co2Color = `#${smokeColor(prediction.mean).getHexString()}`

  return (
    <div className="app">
      <a className="skip-link" href="#controls-title">
        Bỏ qua tới bảng thông số
      </a>
      <header className="topbar">
        <a className="brand" href="./">
          <span className="brand-dot" aria-hidden="true" />
          CO₂ Studio
        </a>
        <nav aria-label="Liên kết" className="topnav mono">
          <a href={REPO}>Mã nguồn</a>
          <a href={`${REPO}/blob/main/data/README.md`}>Dữ liệu</a>
          <a href={`${REPO}/blob/main/docs/decisions.md`}>Quyết định</a>
        </nav>
      </header>

      <main className="page">
        <header className="intro" data-enter>
          <p className="eyebrow">
            <span className="accent-orange">Lab 01</span> · Hồi quy tuyến tính · Khoa CNTT · HCMUS
          </p>
          <h1>
            Xưởng xe 3D: mỗi thông số là một bộ phận, <span className="accent-cyan">khói xả là CO₂</span>
          </h1>
          <p className="lede">
            Mô hình {model.name} học từ 1.067 xe hạng nhẹ Canada năm 2014 và tính lượng CO₂ (g/km) ngay trên trình duyệt. Đổi
            một thông số: bộ phận đó sáng lên, camera bay tới, nhãn nổi báo CO₂ tăng hay giảm bao nhiêu gam. Dưới sàn là cả đội
            xe 2014 cuộn thành vòng đo, dải cam là khoảng dự đoán 95%.
          </p>
        </header>

        <div className="workshop">
          <Controls config={config} bundle={bundle} thumbs={thumbs} onChange={onChange} onPreset={onPreset} />

          <div className="hud stage">
            <Viewport
              config={config}
              prediction={prediction}
              fleet={bundle.fleet}
              change={change}
              mode={tab === 'lineup' ? 'lineup' : 'single'}
              xray={tab === 'xray'}
              pixelSize={pixelSize}
              follow={follow}
              autoRotate={autoRotate}
              view={view}
              lineup={lineup}
              onSelectBody={onSelectBody}
              onThumbnails={setThumbs}
            />

            <div className="stage-top">
              <div className="readout" aria-live="polite">
                <p className="readout-value" style={{ color: co2Color }}>
                  {fmt(prediction.mean, 1)}
                  <span className="readout-unit">g/km CO₂</span>
                </p>
                <p className="readout-sub mono">
                  Một chiếc cụ thể: {fmt(prediction.pi[0], 1)} – {fmt(prediction.pi[1], 1)} · nhiều hơn{' '}
                  {Math.round(prediction.percentile * 100)}% đội xe
                </p>
              </div>
              <div className="seg-tabs" role="group" aria-label="Chế độ xem">
                {(
                  [
                    ['studio', 'Studio'],
                    ['xray', 'X-ray'],
                    ['lineup', 'Đội hình 5 xe'],
                  ] as [Tab, string][]
                ).map(([id, label]) => (
                  <button key={id} type="button" className="tab" aria-pressed={tab === id} onClick={() => setTab(id)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="stage-bottom">
              <div className="chips" role="group" aria-label="Góc camera">
                {VIEWS.map((v) => (
                  <button key={v.id} type="button" className="chip" onClick={() => requestView(v.id)} disabled={tab === 'lineup'}>
                    {v.label}
                  </button>
                ))}
              </div>
              <div className="stage-options">
                <div className="seg-tabs pixel" role="group" aria-label="Cỡ điểm ảnh">
                  <span className="seg-label mono">Pixel</span>
                  {PIXEL_SIZES.map((p) => (
                    <button key={p} type="button" className="tab mono" aria-pressed={pixelSize === p} onClick={() => setPixelSize(p)}>
                      {p === 1 ? 'Tắt' : `${p}px`}
                    </button>
                  ))}
                </div>
                <label className="check mono">
                  <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
                  Camera tự theo bộ phận
                </label>
                <label className="check mono">
                  <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} />
                  Tự xoay
                </label>
              </div>
            </div>
          </div>
        </div>

        <div className="cards">
          <ResultCard prediction={prediction} model={model} models={Object.keys(bundle.models) as ModelName[]} onModel={setModelName} />
          <WhyCard prediction={prediction} />
          <LogCard entries={log} />
        </div>
      </main>

      <footer className="footer mono">
        Mission LR-01 · Nhóm 24127020 · 24127039 · 24127183 · 24127500 · 24127508 · GVHD: Thầy Bùi Tiến Lên, Thầy Võ Nhật Tân ·
        Khoa CNTT, HCMUS · Mẫu xe: Kenney Car Kit (CC0) · Dữ liệu: Natural Resources Canada, Fuel Consumption Ratings 2014
      </footer>
    </div>
  )
}
