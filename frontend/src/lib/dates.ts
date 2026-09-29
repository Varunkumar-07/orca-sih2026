// Calendar dates as the user sees them. Date#toISOString() is UTC, so for
// an Indian user between 00:00 and 05:30 IST it still reports yesterday —
// every page's default range then ended a day early and hid that night's
// rows. These read the browser's own local calendar instead.

/** YYYY-MM-DD of `d` in the browser's local timezone. */
export function localIsoDate(d: Date = new Date()): string {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

/** Local calendar date `days` days before `now` (calendar days, not 24h
 * blocks, so a DST change can't shift it). */
export function localIsoDateDaysAgo(days: number, now: Date = new Date()): string {
  const d = new Date(now)
  d.setDate(d.getDate() - days)
  return localIsoDate(d)
}

/** Minutes the browser's timezone is ahead of UTC (IST = 330) — lets the
 * backend read a date range as the same local days the user picked. */
export function localUtcOffsetMinutes(d: Date = new Date()): number {
  return -d.getTimezoneOffset()
}
