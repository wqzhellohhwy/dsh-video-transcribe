/**
 * dsh-video-transcribe 端到端验证。
 *
 * 三段:
 *   1. **真实转写** —— 走工具自己的 `execute({action:'start'})` / `status`,
 *      不碰任何手敲命令,验证「插件被当作工具调用时」的真实行为;
 *   2. **长音频分片断言** —— 时长 > 5 分钟时必须走分片路径,且文稿量要达到健康
 *      水平(修复前 69 分钟那集产出 0 字);
 *   3. **坏片段韧性** —— 片段列表里塞一个不存在的文件,整篇文稿不能被毁掉,
 *      也不能静默少一段。
 *
 * 用法:`node e2e.mjs [url]`(缺省用修复前会返回空文本的那个 69 分钟视频)
 */

import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  __test,
  createRegistry,
  createTool,
  resolveConfig,
  transcribeFiles,
} from './index.js'

/** 修复前:单遍转写返回 transcribe_empty(退出码 0),用户实跑踩到的就是它。 */
const DEFAULT_URL = 'https://www.bilibili.com/video/BV16VHm6sEYB'

const url = process.argv[2] ?? DEFAULT_URL

const cfg = resolveConfig({})
const logger = {
  info: (message) => console.log(`[info] ${message}`),
  warn: (message) => console.log(`[warn] ${message}`),
}

const checks = []
/** 记一条断言,不中断 —— 一次跑完看到全部结论。 */
function check(title, ok, detail = '') {
  checks.push({ title, ok, detail })
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${title}${ok || detail === '' ? '' : `\n       ${detail}`}`)
}

const registry = createRegistry(cfg, logger)
const tool = createTool(cfg, registry, logger)

console.log(`工具:${tool.name}`)
console.log(`URL:${url}`)
console.log(`作业根目录:${cfg.jobsDir}\n`)

/* -------------------------------------------------------------------------- */
/* 1. 真实转写                                                                 */
/* -------------------------------------------------------------------------- */

const started = await tool.execute({ action: 'start', url, waitSeconds: 0 }, {})
if (started.error !== undefined) {
  console.error(`启动失败:${started.error}`)
  process.exit(1)
}

const jobId = started.jobId
console.log(`作业已建:${jobId}(状态 ${started.status})\n`)

const deadline = Date.now() + 40 * 60_000
let lastPhase = ''
let final = undefined

while (Date.now() < deadline) {
  await new Promise((resolve) => setTimeout(resolve, 10_000))
  const status = await tool.execute({ action: 'status', jobId }, {})
  if (status.error !== undefined) {
    console.error(`查询失败:${status.error}`)
    process.exit(1)
  }
  if (status.phase !== lastPhase) {
    lastPhase = status.phase
    console.log(`阶段 → ${status.phase}(已用 ${Math.round((status.elapsedMs ?? 0) / 1000)}s)`)
  }
  if (['done', 'failed', 'cancelled'].includes(status.status)) {
    final = status
    break
  }
}

if (final === undefined) {
  console.error('等待超时')
  process.exit(1)
}

console.log(`\n最终状态:${final.status}`)
const blocks = tool.output.render({}, { action: 'status', ...final })
const rendered = blocks[0].text
console.log('--- render 输出(前 1200 字符)---')
console.log(rendered.slice(0, 1200))
console.log(`\n--- render 输出总长:${rendered.length} 字符 ---`)
console.log(`文稿文件:${final.textPath ?? '(无)'}\n`)

check('转写成功', final.status === 'done', `status=${final.status} ${final.error ?? ''}`)
if (final.status !== 'done') {
  console.log('\n转写未成功,后续断言跳过。')
  console.log(`\n${checks.filter((c) => c.ok).length}/${checks.length} 通过`)
  process.exit(1)
}

const job = await registry.get(jobId)
const durationSec = job.meta?.durationSec ?? 0

/* -------------------------------------------------------------------------- */
/* 2. 长音频必须分片,且文稿量要健康                                            */
/* -------------------------------------------------------------------------- */

console.log('分片断言:')
if (durationSec > 300) {
  const segmentCount = job.segments?.length ?? job.segmentPlan?.length ?? 0
  check('长音频走了分片路径', job.chunked === true)
  const expected = Math.floor(durationSec / cfg.chunkSeconds)
  check(
    `片数与时长匹配(${segmentCount} 片,期望约 ${expected} 片)`,
    segmentCount >= expected - 5 && segmentCount <= expected + 5,
    `chunkSeconds=${cfg.chunkSeconds}`,
  )
  check(`文稿量达到健康水平(${job.charCount} 字符 > 4000)`, job.charCount > 4000, `修复前为 0 字符`)
  check('返回里带每片字数文件', typeof job.segmentsPath === 'string' && existsSync(job.segmentsPath))
  check('片段文件已清理(不留几百 MB 垃圾)', !existsSync(join(job.dir, 'chunks')))
  check('产出密度不告警(≥1 字/秒)', !(job.warnings ?? []).some((w) => w.includes('产出密度')), (job.warnings ?? []).join(' / '))

  if (typeof job.segmentsPath === 'string' && existsSync(job.segmentsPath)) {
    const segments = JSON.parse(readFileSync(job.segmentsPath, 'utf8'))
    const covered = segments.reduce((sum, segment) => sum + (segment.chars ?? 0), 0)
    check(
      `每片字数之和与文稿量级一致(${covered} vs ${job.charCount})`,
      covered >= job.charCount * 0.9 && covered <= job.charCount * 1.5,
    )
    const withText = segments.filter((segment) => (segment.chars ?? 0) > 0).length
    check(`绝大多数片段都有文字(${withText}/${segments.length})`, withText >= segments.length * 0.8)
  }
} else {
  check('短视频不必分片', job.chunked !== true)
}

/* -------------------------------------------------------------------------- */
/* 3. 坏片段韧性                                                               */
/* -------------------------------------------------------------------------- */

console.log('\n坏片段韧性:')
const scratch = mkdtempSync(join(tmpdir(), 'vt-e2e-'))
const clips = []
for (const [index, start] of [0, 60].entries()) {
  const out = join(scratch, `clip-${index}.wav`)
  const result = await __test.run(cfg.ffmpeg, [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-ss', String(start), '-t', '20', '-i', job.wavFile,
    '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', out,
  ])
  if (result.code === 0) clips.push(out)
}

if (clips.length === 2) {
  const outcome = await transcribeFiles(
    cfg,
    [clips[0], join(scratch, 'does-not-exist.wav'), clips[1]],
    { language: job.language, logger },
  )
  check('整体仍算成功', outcome.ok === true, JSON.stringify(outcome.message ?? ''))
  check('段落数与输入文件数一致(坏的那段留空,而不是少一段)', outcome.texts.length === 3, `texts=${outcome.texts?.length}`)
  check('坏的那段是空串', outcome.texts?.[1] === '', JSON.stringify(outcome.texts?.[1] ?? null))
  check('好片段有文字', (outcome.texts?.[0]?.length ?? 0) > 0 && (outcome.texts?.[2]?.length ?? 0) > 0)
  check('原始输出被留档', typeof outcome.raws?.join('') === 'string' && outcome.raws.join('').length > 0)
} else {
  check('准备测试片段', false, 'ffmpeg 切片段失败')
}

/* -------------------------------------------------------------------------- */

const passed = checks.filter((c) => c.ok).length
console.log(`\n${passed}/${checks.length} 通过`)
process.exit(passed === checks.length ? 0 : 1)
