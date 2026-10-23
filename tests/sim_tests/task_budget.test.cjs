// 在仓库根目录运行：node --test tests/sim_tests/task_budget.test.cjs
// 两个页面执行真实脚本；网络由夹具接管，不连接设备或提交真实任务。
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const flush = () => new Promise(resolve => setImmediate(resolve));

async function page(surface, configuredMode) {
  const web = path.join(__dirname, '../../src/navigation_agent/app/web', surface);
  const html = readFileSync(path.join(web, 'index.html'), 'utf8');
  const script = readFileSync(path.join(web, 'app.js'), 'utf8');
  const elements = new Map(), requests = [];
  let generated = 0;
  function element(id) {
    if (!elements.has(id)) {
      const tag = html.match(new RegExp('<[^>]*\\bid="' + id + '"[^>]*>'))?.[0] || '';
      elements.set(id, {
        value: '', checked: /\bchecked\b/.test(tag),
        hidden: /\bhidden\b/.test(tag), disabled: /\bdisabled\b/.test(tag),
        handlers: {}, style: {}, dataset: {}, options: [], children: [],
        classList: {toggle() {}, add() {}, remove() {}},
        querySelector: () => element(id + '-child'),
        addEventListener(event, handler) { this.handlers[event] = handler; },
        setAttribute() {}, removeAttribute() {}, getContext() { return {}; },
        replaceChildren() {}, append() {}, reportValidity() { return true; },
      });
    }
    return elements.get(id);
  }
  const reply = data => ({ok: true, json: async () => data});
  const context = vm.createContext({
    document: {
      getElementById: element, querySelectorAll: () => [], addEventListener() {},
      createElement: () => element('generated-' + generated++),
      createTextNode: text => ({textContent: text}), createDocumentFragment: () => element('fragment'),
    },
    window: {addEventListener() {}}, location: {protocol: 'http:', host: 'localhost'},
    WebSocket: class {close() {}}, AbortSignal: {timeout() {}},
    localStorage: {getItem() { return ''; }},
    setTimeout() {}, clearTimeout() {}, setInterval() {},
    async fetch(url, options = {}) {
      const endpoint = url.replace('/api/agent', '');
      if (options.method === 'POST') {
        requests.push({url, body: JSON.parse(options.body)});
        return {ok: false, status: 503, json: async () => ({detail: '测试请求已截获'})};
      }
      if (endpoint === '/settings/agent') return reply({
        navigation_time_limit_enabled: configuredMode, navigation_seconds: 27,
        ui_task_defaults: {max_steps: 30, max_seconds: 600},
        task_limits: {min_seconds: 10, max_seconds: 900,
          min_navigation_seconds: 1, max_navigation_seconds: 60},
      });
      if (endpoint === '/tasks' || endpoint.startsWith('/tasks?')) return reply({tasks: []});
      // 运行参数服务不可用不应影响已有任务预算配置。
      return {ok: false, status: 503, json: async () => ({})};
    },
  });
  vm.runInContext(script, context);
  await flush();
  return {element, requests, async toggle(enabled) {
    element('navigation-time-limit-enabled').checked = enabled;
    await element('navigation-time-limit-enabled').handlers.change();
    await flush();
  }, async submit() {
    element(surface === 'monitor' ? 'goal-input' : 'goal').value = '找到沙发';
    const form = element('task-form');
    await (form.onsubmit || form.handlers.submit)({preventDefault() {}});
    await flush();
  }};
}

for (const surface of ['mujoco', 'monitor']) {
  for (const configuredMode of [false, true]) {
    test(`${surface} 页面默认关闭，优先于配置 ${configuredMode}，不提交残留秒数`, async () => {
      const ui = await page(surface, configuredMode);
      assert.equal(ui.element('navigation-time-limit-enabled').checked, false);
      assert.equal(ui.element('navigation-limit').hidden, true);
      assert.equal(ui.element('navigation-seconds').disabled, true);
      assert.equal(ui.element('navigation-unlimited').hidden, false);
      ui.element('navigation-seconds').value = '15';
      await ui.submit();
      assert.equal(ui.requests.length, 1);
      assert.deepEqual(ui.requests[0].body, {
        goal: '找到沙发', max_steps: 30, max_seconds: 600, navigation_time_limit_enabled: false,
      });
    });
  }
  test(`${surface} 显式开启后提交用户秒数，不写运行设置`, async () => {
    const ui = await page(surface, false);
    await ui.toggle(true);
    assert.equal(ui.element('navigation-limit').hidden, false);
    assert.equal(ui.element('navigation-seconds').disabled, false);
    assert.equal(ui.element('navigation-unlimited').hidden, true);
    assert.equal(ui.element('navigation-seconds').value, 27);
    ui.element('navigation-seconds').value = '32';
    if (surface === 'monitor') {
      ui.element('navigation-seconds').handlers.input();
      await ui.element('settings-load').handlers.click();
      assert.equal(ui.element('navigation-time-limit-enabled').checked, true);
      assert.equal(ui.element('navigation-seconds').value, '32');
    }
    await ui.submit();
    assert.equal(ui.requests.length, 1);
    assert.ok(ui.requests[0].url.endsWith('/tasks'));
    assert.equal(ui.requests[0].body.navigation_seconds, 32);
    assert.equal(ui.requests[0].body.navigation_time_limit_enabled, true);
  });
  test(`${surface} 再次关闭后不发送旧秒数`, async () => {
    const ui = await page(surface, true);
    await ui.toggle(true);
    ui.element('navigation-seconds').value = '32';
    await ui.toggle(false);
    await ui.submit();
    assert.equal(ui.requests.length, 1);
    assert.equal(ui.requests[0].body.navigation_time_limit_enabled, false);
    assert.equal('navigation_seconds' in ui.requests[0].body, false);
  });
}
