import type {Observation} from './api';
import {inputObservation, retainedObservation} from './featureTypes';

// A Track must contain both exact endpoints in the same observation source.
export function commonMatchTracks(left: Observation[], right: Observation[], a: number, b: number): string[] {
  const rightByTrack = new Map<string, Observation[]>();
  for (const observation of right) {
    if (observation.feature !== b) continue;
    const entries = rightByTrack.get(observation.track) || [];
    entries.push(observation);
    rightByTrack.set(observation.track, entries);
  }
  const result = new Set<string>();
  for (const observation of left) {
    if (observation.feature !== a) continue;
    if (rightByTrack.get(observation.track)?.some(other =>
      (inputObservation(observation) && inputObservation(other)) ||
      (retainedObservation(observation) && retainedObservation(other)))) {
      result.add(observation.track);
    }
  }
  return [...result].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
}
