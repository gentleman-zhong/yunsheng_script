import {useEffect, useRef, type ReactNode} from 'react';
import {createPortal} from 'react-dom';

const dialogs: HTMLElement[] = [];
let previousOverflow = '';

export function Dialog({title, onClose, children}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const previous = document.activeElement as HTMLElement | null;
    if (!dialogs.length) {
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
    }
    dialogs.push(element);
    element.focus();

    const key = (event: KeyboardEvent) => {
      if (dialogs[dialogs.length - 1] !== element) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (document.fullscreenElement) void fullscreen(element);
        else close.current();
      }
      if (event.key !== 'Tab') return;
      const nodes = Array.from(element.querySelectorAll<HTMLElement>(
        'button, input, select, textarea, a[href], summary, [tabindex]'
      )).filter(node => node.tabIndex >= 0 && !node.matches(':disabled') &&
        node.getClientRects().length > 0 && getComputedStyle(node).visibility !== 'hidden');
      if (!nodes.length) {
        event.preventDefault();
        element.focus();
        return;
      }
      const first = nodes[0], last = nodes[nodes.length - 1];
      const outside = !nodes.includes(document.activeElement as HTMLElement);
      if (event.shiftKey && (document.activeElement === first || outside)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || outside)) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', key);
    return () => {
      const wasTop = dialogs[dialogs.length - 1] === element;
      const index = dialogs.indexOf(element);
      if (index >= 0) dialogs.splice(index, 1);
      if (!dialogs.length) document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', key);
      if (wasTop) {
        if (previous?.isConnected) previous.focus();
        else dialogs[dialogs.length - 1]?.focus();
      }
    };
  }, []);

  return createPortal(<div className="viewer-dialog" ref={ref} tabIndex={-1}
    role="dialog" aria-modal="true" aria-label={title}>{children}</div>, document.body);
}

export async function fullscreen(element: HTMLElement | null) {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (element?.requestFullscreen) await element.requestFullscreen();
  } catch {
    // The full-window dialog remains available when fullscreen is unavailable.
  }
}
