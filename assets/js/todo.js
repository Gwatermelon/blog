(() => {
  'use strict';

  const stateKey = 'gz-research-todo-v2';
  const customTasksKey = 'gz-research-custom-tasks-v1';
  const legacyNotesKey = 'gz-research-notes-v1';
  const groups = [...document.querySelectorAll('[data-todo-group]')];
  const progressText = document.getElementById('todo-progress-text');
  const progressBar = document.getElementById('todo-progress-bar');
  const form = document.getElementById('todo-capture-form');
  const taskText = document.getElementById('new-todo-text');
  const taskNote = document.getElementById('new-todo-note');
  const formStatus = document.getElementById('todo-form-status');
  const resetButton = document.getElementById('todo-reset');

  if (!progressText || !progressBar || !form || !taskText || !taskNote || !formStatus || !resetButton) {
    return;
  }

  function setStorageWarning() {
    formStatus.textContent = '浏览器存储不可用，本次修改仅在当前页面有效';
  }

  function readStorage(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : JSON.parse(value);
    } catch (_) {
      setStorageWarning();
      return fallback;
    }
  }

  function writeStorage(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (_) {
      setStorageWarning();
      return false;
    }
  }

  function removeStorage(key) {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (_) {
      setStorageWarning();
      return false;
    }
  }

  let saved = readStorage(stateKey, {});
  let customTasks = readStorage(customTasksKey, []);
  if (!saved || typeof saved !== 'object' || Array.isArray(saved)) saved = {};
  if (!Array.isArray(customTasks)) customTasks = [];

  customTasks = customTasks.filter(task => (
    task &&
    typeof task.id === 'string' &&
    typeof task.text === 'string' &&
    Number.isInteger(Number(task.groupIndex))
  ));

  function allBoxes() {
    return [...document.querySelectorAll('[data-task-id]')];
  }

  function saveCustomTasks() {
    return writeStorage(customTasksKey, customTasks);
  }

  function saveCheckedState() {
    return writeStorage(stateKey, saved);
  }

  function updateTaskNumbers() {
    document.querySelectorAll('.todo-item-row').forEach((row, index) => {
      const number = row.querySelector('.todo-number');
      if (number) number.textContent = index + 1;
    });
  }

  function updateProgress() {
    const boxes = allBoxes();
    const done = boxes.filter(box => box.checked).length;
    progressText.textContent = `${done} / ${boxes.length}`;
    progressBar.style.width = boxes.length ? `${done / boxes.length * 100}%` : '0%';
    updateTaskNumbers();
  }

  function bindBox(box) {
    box.checked = Boolean(saved[box.dataset.taskId]);
    box.addEventListener('change', () => {
      saved[box.dataset.taskId] = box.checked;
      saveCheckedState();
      updateProgress();
    });
  }

  function buildCustomTask(task) {
    const row = document.createElement('div');
    row.className = 'todo-item-row todo-item-custom';
    row.dataset.customTaskId = task.id;

    const label = document.createElement('label');
    label.className = 'todo-item';
    label.htmlFor = `task-${task.id}`;

    const box = document.createElement('input');
    box.type = 'checkbox';
    box.id = `task-${task.id}`;
    box.dataset.taskId = task.id;

    const check = document.createElement('span');
    check.className = 'todo-check';
    check.setAttribute('aria-hidden', 'true');

    const number = document.createElement('span');
    number.className = 'todo-number';
    number.setAttribute('aria-hidden', 'true');

    const copy = document.createElement('span');
    copy.className = 'todo-copy';
    const title = document.createElement('strong');
    title.textContent = task.text;
    copy.appendChild(title);
    if (task.note) {
      const note = document.createElement('small');
      note.textContent = task.note;
      copy.appendChild(note);
    }

    label.append(box, check, number, copy);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'todo-delete';
    remove.textContent = '删除';
    remove.setAttribute('aria-label', `删除事项：${task.text}`);
    remove.addEventListener('click', () => {
      customTasks = customTasks.filter(item => item.id !== task.id);
      delete saved[task.id];
      const tasksSaved = saveCustomTasks();
      const checksSaved = saveCheckedState();
      row.remove();
      formStatus.textContent = tasksSaved && checksSaved
        ? '事项已删除'
        : '事项已从当前页面删除，但浏览器存储不可用';
      updateProgress();
    });

    row.append(label, remove);
    bindBox(box);
    return row;
  }

  document.querySelectorAll('[data-task-id]').forEach(bindBox);

  customTasks.forEach(task => {
    const group = groups[Number(task.groupIndex)];
    if (group) group.querySelector('[data-todo-items]').appendChild(buildCustomTask(task));
  });

  let legacyNotes = '';
  try {
    legacyNotes = localStorage.getItem(legacyNotesKey) || '';
  } catch (_) {
    setStorageWarning();
  }
  if (legacyNotes && !taskText.value) {
    taskText.value = legacyNotes;
    formStatus.textContent = '已载入之前的临时研究笔记，请选择分类后添加';
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    const text = taskText.value.trim();
    const note = taskNote.value.trim();
    const selected = form.querySelector('input[name="todo-group"]:checked');
    const groupIndex = selected ? Number(selected.value) : 0;
    const group = groups[groupIndex];

    if (!text) {
      formStatus.textContent = '请先填写要研究的内容';
      taskText.focus();
      return;
    }
    if (!group) {
      formStatus.textContent = '请选择有效的事项类型';
      return;
    }

    const randomPart = window.crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
    const task = {
      id: `custom-${Date.now()}-${randomPart}`,
      groupIndex,
      text,
      note,
      createdAt: new Date().toISOString()
    };

    customTasks.push(task);
    const taskSaved = saveCustomTasks();
    group.querySelector('[data-todo-items]').appendChild(buildCustomTask(task));
    removeStorage(legacyNotesKey);
    taskText.value = '';
    taskNote.value = '';
    formStatus.textContent = taskSaved
      ? `已添加到「${group.querySelector('h2').textContent}」`
      : '事项已添加，但浏览器存储不可用，本次仅在当前页面有效';
    taskText.focus();
    updateProgress();
  });

  resetButton.addEventListener('click', () => {
    allBoxes().forEach(box => { box.checked = false; });
    saved = {};
    const stateRemoved = removeStorage(stateKey);
    formStatus.textContent = stateRemoved
      ? '全部勾选状态已重置'
      : '当前页面状态已重置，但浏览器存储不可用';
    updateProgress();
  });

  updateProgress();
})();
