// 各可视化页面共用的顶部导航和亮色 / 暗色切换。
// 用法：页面里放 <nav class="sitenav"></nav>，再引入本文件。加 data-floating 时浮在页面右上角（3D 页面用）。
(() => {
  const PAGES = [
    ['overview.html', '总流程'],
    ['progress.html', '全部工作'],
    ['pipeline.html', '工作流程'],
    ['forgetting.html', '遗忘曲线'],
    ['demo.html', '概率海演示'],
    ['live.html', '实时数据'],
  ];
  const here = location.pathname.split('/').pop() || 'overview.html';

  const css = `
  nav.sitenav { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 2px; margin: 0 0 10px;
    font: 13px/1.4 "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans SC", system-ui, sans-serif; }
  nav.sitenav a { color: var(--ink-2, #52514e); text-decoration: none; padding: 3px 10px; border-radius: 999px;
    border: 1px solid transparent; }
  nav.sitenav a:hover { border-color: var(--border, rgba(0,0,0,.12)); color: var(--ink, #0b0b0b); }
  nav.sitenav a[aria-current] { color: var(--ink, #0b0b0b); font-weight: 600;
    background: var(--surface, #fff); border-color: var(--border, rgba(0,0,0,.12)); }
  nav.sitenav .brand { color: var(--muted, #898781); margin-right: 8px; font-size: 12px; }
  nav.sitenav .theme { margin-left: auto; font: inherit; font-size: 12px; cursor: pointer; color: var(--ink-2, #52514e);
    background: transparent; border: 1px solid var(--border, rgba(0,0,0,.12)); border-radius: 999px; padding: 2px 10px; }
  nav.sitenav[data-floating] { position: fixed; z-index: 5; top: 10px; right: 12px; margin: 0;
    background: rgba(20,24,33,.85); padding: 4px 6px; border-radius: 999px; }
  nav.sitenav[data-floating] a { color: #9aa3b5; }
  nav.sitenav[data-floating] a[aria-current] { color: #fff; background: rgba(255,255,255,.08); border-color: transparent; }
  nav.sitenav[data-floating] .brand, nav.sitenav[data-floating] .theme { display: none; }
  `;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  // 主题：auto 跟随系统；light / dark 写到 <html data-theme>，各页面的 CSS 变量据此切换
  const KEY = 'viz-theme';
  const read = () => { try { return localStorage.getItem(KEY) || 'auto'; } catch { return 'auto'; } };
  const apply = t => {
    if (t === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
  };
  apply(read());
  const LABEL = { auto: '主题：跟随系统', light: '主题：亮色', dark: '主题：暗色' };
  const NEXT = { auto: 'light', light: 'dark', dark: 'auto' };

  for (const nav of document.querySelectorAll('nav.sitenav')) {
    nav.textContent = '';
    const brand = document.createElement('span');
    brand.className = 'brand';
    brand.textContent = '类人 AI · 可视化';
    nav.appendChild(brand);
    for (const [href, name] of PAGES) {
      const a = document.createElement('a');
      a.href = href;
      a.textContent = name;
      if (href === here) a.setAttribute('aria-current', 'page');
      nav.appendChild(a);
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'theme';
    btn.textContent = LABEL[read()];
    btn.onclick = () => {
      const t = NEXT[read()];
      try { localStorage.setItem(KEY, t); } catch { /* 无痕窗口等情况存不了，只对本次生效 */ }
      apply(t);
      btn.textContent = LABEL[t];
      document.dispatchEvent(new CustomEvent('viz-theme'));
    };
    nav.appendChild(btn);
  }
})();
