import { useEffect, useRef, useState } from 'react'
import type { Body, Config, Prediction } from '../model'
import { Studio, type Part } from '../scene/Studio'

export interface Change {
  id: number
  part: Part
  delta: number
}

interface Props {
  config: Config
  prediction: Prediction
  fleet: number[]
  change: Change | null
  mode: 'single' | 'lineup'
  pixelSize: number
  xray: boolean
  lineup: { configs: Config[]; co2: number[] }
  onSelectBody: (body: Body) => void
}

export function Viewport(props: Props) {
  const host = useRef<HTMLDivElement>(null)
  const studio = useRef<Studio | null>(null)
  const lastChange = useRef(0)
  const [state, setState] = useState<'loading' | 'ready' | 'nogl' | 'error'>(() => (Studio.supported() ? 'loading' : 'nogl'))
  const selectBody = useRef(props.onSelectBody)
  selectBody.current = props.onSelectBody

  useEffect(() => {
    if (!host.current || state === 'nogl') return
    let cancelled = false
    let instance: Studio
    try {
      instance = new Studio(host.current, { onSelectBody: (b) => selectBody.current(b) })
    } catch {
      setState('nogl')
      return
    }
    instance.setFleet(props.fleet)
    instance
      .load(import.meta.env.BASE_URL)
      .then(() => {
        if (cancelled) return
        studio.current = instance
        setState('ready')
      })
      .catch(() => !cancelled && setState('error'))
    return () => {
      cancelled = true
      studio.current = null
      instance.dispose()
    }
    // The studio lives for the lifetime of the component.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const s = studio.current
    if (!s) return
    const change = props.change && props.change.id !== lastChange.current ? props.change : undefined
    if (change) lastChange.current = change.id
    s.update(props.config, props.prediction.mean, change)
    s.setPiBand(props.prediction.pi[0], props.prediction.pi[1])
    s.updateLineup(props.lineup.configs, props.lineup.co2, props.config.BODY)
  }, [state, props.config, props.prediction, props.change, props.lineup])

  useEffect(() => studio.current?.setMode(props.mode), [state, props.mode])
  useEffect(() => studio.current?.setPixelSize(props.pixelSize), [state, props.pixelSize])
  useEffect(() => studio.current?.setXray(props.xray), [state, props.xray])

  return (
    <div className="viewport" aria-label="Xưởng xe 3D: mô hình xe thay đổi theo thông số" role="img">
      <div ref={host} className="viewport-canvas" />
      {state === 'loading' && <p className="viewport-note">Đang nạp mẫu xe…</p>}
      {state === 'error' && <p className="viewport-note">Không nạp được mẫu xe. Tải lại trang để thử lại.</p>}
      {state === 'nogl' && (
        <p className="viewport-note">
          Trình duyệt này không bật WebGL nên không vẽ được xưởng 3D. Mọi con số bên phải vẫn tính đúng; bật tăng tốc phần
          cứng để xem mô hình xe.
        </p>
      )}
    </div>
  )
}
