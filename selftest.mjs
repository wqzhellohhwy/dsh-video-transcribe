/**
 * dsh-video-transcribe 自测。
 *
 * 覆盖两类东西:
 *   1. **纯函数** —— 配置解析、URL 闸门、sherpa stdout 解析;
 *   2. **注册契约** —— 经真实 `apply` + 假 `ctx.tools` 捕获工具定义,断言
 *      `name / description / parameters / execute / output` 齐全(宿主缺 `output`
 *      会拒绝注册,单测假 register 不校验,所以这里自己校),并逐个走一遍
 *      execute 的错误分支与 render 的全函数性。
 *
 * 不联网、不起子进程:端到端另有真实验证。
 *
 * 用法:`node selftest.mjs`
 */

import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  __test,
  apply,
  compact,
  countWords,
  createRegistry,
  extractTranscript,
  formatResult,
  isHttpUrl,
  resolveConfig,
  splitTranscripts,
  yieldWarnings,
} from './index.js'
import { buildPcmWav, joinSegments, parsePcmWav, planSegments, readWavInfo } from './segment.js'

let passed = 0
const cases = []

/** 跑一个用例,失败就记下来(全跑完再汇总,便于一次看到全部问题)。 */
async function test(title, fn) {
  try {
    await fn()
    passed += 1
    cases.push(`  ok   ${title}`)
  } catch (error) {
    cases.push(`  FAIL ${title}\n       ${String(error?.message ?? error)}`)
  }
}

const silentLogger = { info() {}, warn() {} }

/* -------------------------------------------------------------------------- */
/* 1. 配置解析                                                                 */
/* -------------------------------------------------------------------------- */

await test('resolveConfig:缺省路径落在 DSH_HOME 下', () => {
  const cfg = resolveConfig(undefined)
  assert.ok(cfg.root.includes('video-transcribe'), `root=${cfg.root}`)
  assert.ok(cfg.ytDlp.endsWith('yt-dlp.exe'), cfg.ytDlp)
  assert.ok(cfg.ffmpeg.endsWith('ffmpeg.exe'), cfg.ffmpeg)
  assert.ok(cfg.sherpa.endsWith('sherpa-onnx-offline.exe'), cfg.sherpa)
  assert.ok(cfg.model.endsWith('model.int8.onnx'), cfg.model)
  assert.ok(cfg.tokens.endsWith('tokens.txt'), cfg.tokens)
  assert.equal(cfg.language, 'auto')
  assert.ok(Number.isInteger(cfg.threads) && cfg.threads >= 2, `threads=${cfg.threads}`)
})

await test('resolveConfig:畸形 config 不产生不可用配置', () => {
  for (const bad of [null, 'x', 42, [], { threads: -5 }, { language: 'klingon' }]) {
    const cfg = resolveConfig(bad)
    assert.ok(cfg.ytDlp.length > 0)
    assert.ok(cfg.threads >= 2, `threads=${cfg.threads}`)
    assert.ok(['auto', 'zh', 'en', 'ja', 'ko', 'yue'].includes(cfg.language))
  }
})

await test('resolveConfig:显式路径被采纳', () => {
  const cfg = resolveConfig({ ytDlpPath: 'D:\\x\\yt-dlp.exe', threads: 3, language: 'en' })
  assert.equal(cfg.ytDlp, 'D:\\x\\yt-dlp.exe')
  assert.equal(cfg.threads, 3)
  assert.equal(cfg.language, 'en')
})

/* -------------------------------------------------------------------------- */
/* 2. URL 闸门                                                                 */
/* -------------------------------------------------------------------------- */

await test('isHttpUrl:只放行 http(s)', () => {
  assert.equal(isHttpUrl('https://www.bilibili.com/video/BV1SRNF6PE4N/'), true)
  assert.equal(isHttpUrl('http://example.com/a'), true)
  assert.equal(isHttpUrl('file:///C:/secret.txt'), false)
  assert.equal(isHttpUrl('ftp://a/b'), false)
  assert.equal(isHttpUrl('javascript:alert(1)'), false)
  assert.equal(isHttpUrl(''), false)
  assert.equal(isHttpUrl(undefined), false)
})

/* -------------------------------------------------------------------------- */
/* 3. sherpa 输出解析                                                          */
/* -------------------------------------------------------------------------- */

await test('extractTranscript:取 text 并解转义', () => {
  const sample = '{"text": "hello \\"world\\" ok", "tokens": [], "words": []}'
  assert.equal(extractTranscript(sample), 'hello "world" ok')
})

await test('extractTranscript:多段拼接、容忍前置噪声', () => {
  const noisy = 'loading model...\n{"text": "第一段", "tokens": []}\n{"text": "第二段"}\n'
  assert.equal(extractTranscript(noisy), '第一段 第二段')
})

await test('extractTranscript:没有 text 字段返回空串', () => {
  assert.equal(extractTranscript('no json here'), '')
  assert.equal(extractTranscript(''), '')
})

/* -------------------------------------------------------------------------- */
/* 4. 注册契约                                                                 */
/* -------------------------------------------------------------------------- */

/** 用假 ctx 跑一次 apply,返回捕获到的工具定义。 */
function captureTool(config) {
  const captured = []
  const ctx = {
    logger: () => silentLogger,
    tools: { register(def) { captured.push(def) } },
  }
  apply(ctx, config)
  return captured
}

await test('apply:注册恰好一个工具,形状齐全', () => {
  const captured = captureTool({})
  assert.equal(captured.length, 1)
  const tool = captured[0]
  assert.equal(tool.name, 'video_transcribe')
  assert.ok(typeof tool.description === 'string' && tool.description.length > 50)
  assert.equal(tool.parameters?.type, 'object')
  assert.ok(Array.isArray(tool.parameters.required))
  // 宿主缺 output 会拒绝注册 —— 这里自己把关
  assert.ok(tool.output !== undefined, 'output 元数据缺失')
  assert.ok(tool.output.schema !== undefined, 'output.schema 缺失')
  assert.equal(typeof tool.output.render, 'function')
  assert.equal(typeof tool.execute, 'function')
})

await test('apply:ctx.tools 缺失时不抛、不注册', () => {
  let threw = false
  try {
    apply({ logger: () => silentLogger }, {})
  } catch {
    threw = true
  }
  assert.equal(threw, false)
})

await test('apply:register 抛错被吞掉,不影响加载', () => {
  let threw = false
  try {
    apply({ logger: () => silentLogger, tools: { register() { throw new Error('host said no') } } }, {})
  } catch {
    threw = true
  }
  assert.equal(threw, false)
})

/* -------------------------------------------------------------------------- */
/* 5. execute 的错误分支                                                       */
/* -------------------------------------------------------------------------- */

const cfg = resolveConfig({})
const tool = captureTool({})[0]
const registry = createRegistry(cfg, silentLogger)
const realTool = __test.createTool(cfg, registry, silentLogger)

await test('execute:未知 action', async () => {
  const result = await realTool.execute({ action: 'nope' }, {})
  assert.equal(result.code, 'bad_action')
})

await test('execute:action 缺失', async () => {
  const result = await realTool.execute({}, {})
  assert.equal(result.code, 'bad_action')
})

await test('execute:start 缺 url', async () => {
  const result = await realTool.execute({ action: 'start' }, {})
  assert.equal(result.code, 'missing_url')
})

await test('execute:start 拒绝非 http(s)', async () => {
  const result = await realTool.execute({ action: 'start', url: 'file:///C:/x.txt' }, {})
  assert.equal(result.code, 'bad_url')
})

await test('execute:status 缺 jobId / 未知 jobId', async () => {
  assert.equal((await realTool.execute({ action: 'status' }, {})).code, 'missing_job_id')
  assert.equal((await realTool.execute({ action: 'status', jobId: 'nope' }, {})).code, 'job_not_found')
})

await test('execute:cancel 缺 jobId', async () => {
  assert.equal((await realTool.execute({ action: 'cancel' }, {})).code, 'missing_job_id')
})

await test('execute:list 返回数组', async () => {
  const result = await realTool.execute({ action: 'list' }, {})
  assert.equal(result.action, 'list')
  assert.ok(Array.isArray(result.jobs))
})

/* -------------------------------------------------------------------------- */
/* 6. render 全函数性                                                          */
/* -------------------------------------------------------------------------- */

await test('render:畸形 value 也返回 text 字符串', () => {
  const values = [
    undefined, null, 0, '', 'str', [], {},
    { error: 'boom', code: 'x' },
    { action: 'list', jobs: [] },
    { action: 'list', jobs: [{ id: 'a', status: 'done', phase: 'done' }] },
    { action: 'cancel', jobId: 'a', cancelled: true },
    { action: 'cancel', jobId: 'a', cancelled: false },
    { jobId: 'a', status: 'queued', phase: 'queued' },
    { jobId: 'a', status: 'running', phase: 'transcribe', elapsedMs: 1234, meta: { durationSec: 100 } },
    { jobId: 'a', status: 'done', phase: 'done', text: 'hello', charCount: 5, wordCount: 1, textPath: 'p' },
    { jobId: 'a', status: 'done', phase: 'done', text: 'x'.repeat(60000), textPath: 'p' },
    { jobId: 'a', status: 'failed', phase: 'failed', error: { code: 'c', message: 'm' } },
    { jobId: 'a', status: 'cancelled', phase: 'cancelled' },
  ]
  for (const value of values) {
    const blocks = realTool.output.render({}, value)
    assert.ok(Array.isArray(blocks) && blocks.length > 0, `render(${JSON.stringify(value)?.slice(0, 40)}) 空`)
    assert.equal(typeof blocks[0].text, 'string', `render 的 text 不是 string`)
    assert.equal(blocks[0].type, 'text')
  }
  assert.equal(typeof formatResult(undefined), 'string')
})

await test('render:长文稿被截断并指出文件', () => {
  const text = 'x'.repeat(60000)
  const blocks = realTool.output.render({}, {
    jobId: 'a', status: 'done', phase: 'done', text, textPath: '/tmp/t.txt', charCount: 60000,
  })
  const out = blocks[0].text
  assert.ok(out.includes('已截断'), '未截断')
  assert.ok(out.length < text.length, '未变短')
  assert.ok(out.includes('/tmp/t.txt'), '未给出文件路径')
})

/* -------------------------------------------------------------------------- */
/* 7. 作业仓库(不触网)                                                        */
/* -------------------------------------------------------------------------- */

await test('registry:cancel 未知作业返回 false', () => {
  assert.equal(registry.cancel('does-not-exist'), false)
})

await test('registry:list 在没有作业时返回数组', async () => {
  const jobs = await registry.list(5)
  assert.ok(Array.isArray(jobs))
})

/* -------------------------------------------------------------------------- */
/* 8. 返回值必须是无损 JSON                                                     */
/* -------------------------------------------------------------------------- */

/**
 * 找出返回值里任何不能无损穿过 JSON 的东西。
 *
 * 这条是真机上打回来才补上的:宿主要求工具返回值是**无损 JSON**,一个值为
 * `undefined` 的键就会让整次调用被判 `value is not lossless JSON` 而拒收 ——
 * 假 `register` 的单测完全覆盖不到这一层。
 *
 * @param value - 待检查的返回值。
 * @param path - 当前位置(报错用)。
 * @returns 第一处问题的说明,干净则 undefined。
 */
function findNonJson(value, path = '$') {
  if (value === undefined) return `${path} 是 undefined`
  if (typeof value === 'number' && !Number.isFinite(value)) return `${path} 是 ${String(value)}`
  if (['function', 'symbol', 'bigint'].includes(typeof value)) return `${path} 是 ${typeof value}`
  if (Array.isArray(value)) {
    for (const [index, item] of value.entries()) {
      const found = findNonJson(item, `${path}[${index}]`)
      if (found !== undefined) return found
    }
    return undefined
  }
  if (value !== null && typeof value === 'object') {
    const proto = Object.getPrototypeOf(value)
    if (proto !== Object.prototype && proto !== null) return `${path} 不是普通对象(${proto?.constructor?.name ?? '?'})`
    for (const [key, item] of Object.entries(value)) {
      const found = findNonJson(item, `${path}.${key}`)
      if (found !== undefined) return found
    }
    return undefined
  }
  return undefined
}

await test('compact:剥掉 undefined,保留其余', () => {
  const input = { a: 1, b: undefined, c: { d: undefined, e: 'x' } }
  assert.deepEqual(compact(input), { a: 1, c: { e: 'x' } })
})

await test('compact:产出无损 JSON', () => {
  const messy = { a: undefined, b: { c: undefined, d: 1 }, e: 'x' }
  const clean = compact(messy)
  assert.equal(findNonJson(clean), undefined)
  assert.deepEqual(clean, { b: { d: 1 }, e: 'x' })
})

await test('execute:每个出口的返回值都是无损 JSON', async () => {
  const outputs = [
    await realTool.execute({ action: 'list' }, {}),
    await realTool.execute({ action: 'cancel', jobId: 'nope' }, {}),
    await realTool.execute({ action: 'status', jobId: 'nope' }, {}),
    await realTool.execute({}, {}),
    await realTool.execute({ action: 'start' }, {}),
    await realTool.execute({ action: 'start', url: 'file:///x' }, {}),
  ]
  for (const output of outputs) {
    const found = findNonJson(output)
    assert.equal(found, undefined, `非无损 JSON:${found} —— ${JSON.stringify(output)?.slice(0, 120)}`)
  }
})

await test('execute:真实作业的 list / status 也是无损 JSON', async () => {
  const listed = await realTool.execute({ action: 'list' }, {})
  assert.equal(findNonJson(listed), undefined, `list 非无损 JSON:${findNonJson(listed)}`)
  // 磁盘上有历史作业时顺带验证 status 的三条分支
  const first = listed.jobs?.[0]
  if (first !== undefined) {
    const status = await realTool.execute({ action: 'status', jobId: first.id }, {})
    assert.equal(findNonJson(status), undefined, `status 非无损 JSON:${findNonJson(status)}`)
    assert.equal(status.jobId, first.id)
  }
})

/* -------------------------------------------------------------------------- */
/* 9. 分片:长音频静默丢内容的修复                                              */
/* -------------------------------------------------------------------------- */

/**
 * 造样本:整段静音,只有 `[loudFrom, loudTo)` 是 440 Hz 正弦。
 * 用来构造「有响亮语音 + 已知位置的静音间隙」的音频。
 */
function makeSamples(seconds, { sampleRate = 16000, loudFrom = 0, loudTo = 0 } = {}) {
  const total = Math.round(seconds * sampleRate)
  const samples = new Int16Array(total)
  for (let index = 0; index < total; index += 1) {
    const t = index / sampleRate
    samples[index] = t >= loudFrom && t < loudTo ? Math.round(8000 * Math.sin(2 * Math.PI * 440 * t)) : 0
  }
  return samples
}

await test('planSegments:短音频不切', () => {
  const samples = makeSamples(20)
  const plan = planSegments({ sampleRate: 16000, samples, targetSec: 30, minSec: 5, snapSec: 1.5 })
  assert.equal(plan.length, 1)
  assert.equal(plan[0].startSample, 0)
  assert.equal(plan[0].endSample, samples.length)
})

await test('planSegments:长音频完整覆盖、每片不短于下限、首尾相接', () => {
  const samples = makeSamples(611)
  const plan = planSegments({ sampleRate: 16000, samples, targetSec: 30, minSec: 5, snapSec: 0 })
  assert.equal(plan[0].startSample, 0)
  assert.equal(plan[plan.length - 1].endSample, samples.length)
  assert.ok(plan.length >= 20, `片数太少:${plan.length}`)
  for (const segment of plan) {
    const length = segment.endSample - segment.startSample
    assert.ok(length >= 5 * 16000, `片段短于下限:${length / 16000}s`)
    assert.ok(length <= 30 * 16000, `片段超过目标:${length / 16000}s`)
  }
  for (let index = 1; index < plan.length; index += 1) {
    assert.equal(plan[index].startSample, plan[index - 1].endSample, '片段之间有缝或重叠')
  }
})

await test('planSegments:切点吸附到静音处', () => {
  // 名义切点在 30.0 s,但 29.0~29.4 s 有一段静音,其余都在响。
  const samples = makeSamples(70, { loudFrom: 0, loudTo: 70 })
  for (let index = Math.round(29.0 * 16000); index < Math.round(29.4 * 16000); index += 1) samples[index] = 0

  const fixed = planSegments({ sampleRate: 16000, samples, targetSec: 30, minSec: 5, snapSec: 0 })
  assert.equal(fixed[0].endSample / 16000, 30, '不吸附时应正好切在 30s')

  const snapped = planSegments({ sampleRate: 16000, samples, targetSec: 30, minSec: 5, snapSec: 1.5 })
  const at = snapped[0].endSample / 16000
  assert.ok(at > 29.0 && at <= 29.4, `吸附后落在 ${at}s,没有进静音区`)
})

await test('planSegments:样本为空也不炸', () => {
  const plan = planSegments({ sampleRate: 16000, samples: new Int16Array(0) })
  assert.ok(Array.isArray(plan))
})

await test('buildPcmWav ↔ parsePcmWav:往返保真', () => {
  const samples = makeSamples(3, { loudFrom: 0, loudTo: 3 })
  const wav = buildPcmWav(samples, 16000)
  assert.equal(wav.toString('latin1', 0, 4), 'RIFF')
  assert.equal(wav.length, 44 + samples.length * 2)

  const back = parsePcmWav(wav)
  assert.equal(back.sampleRate, 16000)
  assert.equal(back.samples.length, samples.length)
  assert.equal(back.durationSec, 3)
  assert.deepEqual(Array.from(back.samples.slice(500, 600)), Array.from(samples.slice(500, 600)))
})

await test('parsePcmWav:data 块前面有 LIST 元数据块(ffmpeg 的产物)也能解析', () => {
  const samples = makeSamples(1)
  const plain = buildPcmWav(samples, 16000)
  const list = Buffer.alloc(20)
  list.write('LIST', 0, 'latin1')
  list.writeUInt32LE(12, 4)
  list.write('INFOISFTabcd', 8, 'latin1')
  // 基础头 12 + fmt 块 24 = 36,把 LIST 塞在 data 之前
  const withList = Buffer.concat([plain.subarray(0, 36), list, plain.subarray(36)])

  const parsed = parsePcmWav(withList)
  assert.equal(parsed.samples.length, samples.length)
  assert.equal(parsed.durationSec, 1)
})

await test('parsePcmWav:拒绝非 16 位 PCM / 非单声道 / 非 RIFF', () => {
  const base = buildPcmWav(makeSamples(1), 16000)

  const stereo = Buffer.from(base)
  stereo.writeUInt16LE(2, 22)
  assert.throws(() => parsePcmWav(stereo), /单声道/)

  const eightBit = Buffer.from(base)
  eightBit.writeUInt16LE(8, 34)
  assert.throws(() => parsePcmWav(eightBit), /16 位/)

  assert.throws(() => parsePcmWav(Buffer.alloc(100)))
  assert.throws(() => parsePcmWav(Buffer.from('short')))
})

await test('readWavInfo:只读文件头就能拿到时长', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'vt-selftest-'))
  const file = join(dir, 'a.wav')
  writeFileSync(file, buildPcmWav(makeSamples(7), 16000))
  const info = await readWavInfo(file)
  assert.equal(info.sampleRate, 16000)
  assert.equal(info.durationSec, 7)
})

await test('joinSegments:中文不补空格、拉丁文补空格', () => {
  assert.equal(joinSegments(['你好，', '世界。']), '你好，世界。')
  assert.equal(joinSegments(['hello', 'world']), 'hello world')
  assert.equal(joinSegments(['中', 'hello']), '中hello')
  assert.equal(joinSegments(['hello', '中']), 'hello中')
  assert.equal(joinSegments(['', '  ', 'x']), 'x')
  assert.equal(joinSegments([]), '')
})

await test('splitTranscripts:一个输入文件一段文稿', () => {
  const two = '{"lang": "<|zh|>", "text": "第一段", "timestamps": []}\n' +
    '{"lang": "<|zh|>", "text": "第二段", "timestamps": []}'
  assert.deepEqual(splitTranscripts(two), ['第一段', '第二段'])
  assert.deepEqual(splitTranscripts('{"lang": "<|zh|>", "text": ""}'), [''])
  assert.deepEqual(splitTranscripts('{"text": "旧格式"}'), ['旧格式'], '应退回整段提取')
  assert.deepEqual(splitTranscripts('nothing here'), [])
})

await test('yieldWarnings:健康文稿不告警、退化文稿告警', () => {
  assert.deepEqual(yieldWarnings({ text: 'x'.repeat(3000), durationSec: 600, segments: [], threshold: 1 }), [])
  assert.deepEqual(yieldWarnings({ text: 'x', durationSec: 20, segments: [], threshold: 1 }), [], '短音频不按密度告警')

  const low = yieldWarnings({ text: 'x'.repeat(173), durationSec: 611, segments: [], threshold: 1 })
  assert.equal(low.length, 1)
  assert.ok(low[0].includes('产出密度'), low[0])

  const silent = yieldWarnings({
    text: 'x'.repeat(3000),
    durationSec: 600,
    segments: [{ index: 3, durationSec: 30, chars: 0 }, { index: 4, durationSec: 30, chars: 200 }],
    threshold: 1,
  })
  assert.equal(silent.length, 1)
  assert.ok(silent[0].includes('3'), silent[0])
})

await test('resolveConfig:分片缺省值与非法值回落', () => {
  const defaults = resolveConfig({})
  assert.equal(defaults.chunkSeconds, 30)
  assert.equal(defaults.minChunkSeconds, 5)
  assert.equal(defaults.snapSeconds, 1.5)
  assert.equal(defaults.segmentBatch, 100)
  assert.equal(defaults.keepSegments, false)
  assert.equal(defaults.lowYieldThreshold, 1)

  assert.equal(resolveConfig({ chunkSeconds: 60 }).chunkSeconds, 60)
  assert.equal(resolveConfig({ snapSeconds: 0 }).snapSeconds, 0, '0 是合法值(关掉吸附)')
  assert.equal(resolveConfig({ snapSeconds: -3 }).snapSeconds, 1.5)
  assert.equal(resolveConfig({ keepSegments: true }).keepSegments, true)
})

await test('sherpaArgs:参数齐全且语言受限', () => {
  const args = __test.sherpaArgs(resolveConfig({}), 'zh')
  assert.equal(args.length, 5)
  assert.ok(args.some((a) => a.startsWith('--sense-voice-model=')), args.join(' '))
  assert.ok(args.includes('--sense-voice-language=zh'), args.join(' '))
  assert.ok(args.some((a) => a.startsWith('--num-threads=')), args.join(' '))
  // 非法语言回落到配置值
  assert.ok(__test.sherpaArgs(resolveConfig({ language: 'en' }), 'klingon').includes('--sense-voice-language=en'))
})

await test('render:分片信息与告警出现在输出里', () => {
  const blocks = realTool.output.render({}, {
    jobId: 'a',
    status: 'done',
    phase: 'done',
    text: '文稿',
    charCount: 2,
    wordCount: 1,
    textPath: 'p',
    segmentsPath: 's',
    chunked: true,
    chunkSeconds: 30,
    segmentCount: 21,
    warnings: ['产出密度只有 0.28 字/秒(低于 1;正常中文约 2~6,英文更高),可能有内容没被识别出来,建议抽查后再引用'],
  })
  const out = blocks[0].text
  assert.ok(out.includes('分片转写:21 片'), out)
  assert.ok(out.includes('⚠️'), out)
  assert.ok(out.includes('s'), '未给出每片字数文件')
})

await test('countWords:中文不给词数(按空白切是假数字),拉丁文才给', () => {
  assert.equal(countWords('大家好，今天我们来聊聊本地大模型'), undefined)
  assert.equal(countWords(''), undefined)
  assert.equal(countWords('   '), undefined)
  assert.equal(countWords('this is a short english sentence'), 6)
  assert.equal(countWords('the model is 千问 and it is fast'), 8, '中英混排但以英文为主仍应给词数')
})

await test('render:中文文稿不显示「词数」这一项', () => {
  const blocks = realTool.output.render({}, {
    jobId: 'a', status: 'done', phase: 'done', text: '中文文稿', charCount: 4, textPath: 'p',
  })
  assert.ok(!blocks[0].text.includes('词数'), blocks[0].text)
})

/* -------------------------------------------------------------------------- */

console.log(cases.join('\n'))
console.log(`\n${passed}/${cases.length} 通过`)
if (passed !== cases.length) process.exitCode = 1
