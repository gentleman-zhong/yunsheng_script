import {useEffect, useState} from 'react';

export function useImage(src: string) {
  const [state, setState] = useState<{
    src: string;
    image: HTMLImageElement | null;
    error: string;
  }>({src, image: null, error: ''});
  useEffect(() => {
    setState({src, image: null, error: ''});
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (!cancelled) setState({src, image, error: ''});
    };
    image.onerror = () => {
      if (!cancelled) setState({src, image: null, error: '图像无法读取'});
    };
    image.src = src;
    return () => {
      cancelled = true;
      image.onload = null;
      image.onerror = null;
      image.src = '';
    };
  }, [src]);
  return state.src === src ? {image: state.image, error: state.error} : {image: null, error: ''};
}

export function xy(point: (number | null)[], width: number, height: number) {
  const scale = Math.max(width, height);
  return [(point[0] ?? NaN) * scale + width / 2 - .5,
    (point[1] ?? NaN) * scale + height / 2 - .5];
}
