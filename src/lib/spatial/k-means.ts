import { GraphicsDevice } from 'playcanvas';

import { KdTree } from './kd-tree';
import { Column, DataTable } from '../data-table';
import { GpuClustering } from '../gpu';
import { logger } from '../utils';

// use floyd's algorithm to pick m unique random indices from 0..n-1
const pickRandomIndices = (n: number, m: number) => {
    const chosen = new Set<number>();
    for (let j = n - m; j < n; j++) {
        const t = Math.floor(Math.random() * (j + 1));
        chosen.add(chosen.has(t) ? j : t);
    }
    return [...chosen];
};

const initializeCentroids = (dataTable: DataTable, centroids: DataTable, row: any) => {
    const indices = pickRandomIndices(dataTable.numRows, centroids.numRows);
    for (let i = 0; i < centroids.numRows; ++i) {
        dataTable.getRow(indices[i], row);
        centroids.setRow(i, row);
    }
};

// in the 1d case we use quantile-based initialization for better handling of skewed data
const initializeCentroids1D = (dataTable: DataTable, centroids: DataTable) => {
    const data = dataTable.getColumn(0).data;
    const n = dataTable.numRows;
    const k = centroids.numRows;

    // Sort data to compute quantiles
    const sorted = Float32Array.from(data).sort((a, b) => a - b);

    const centroidsData = centroids.getColumn(0).data;
    for (let i = 0; i < k; ++i) {
        // Place centroid at the center of its expected cluster region
        const quantile = (2 * i + 1) / (2 * k);
        const index = Math.min(Math.floor(quantile * n), n - 1);
        centroidsData[i] = sorted[index];
    }
};

const clusterKdTreeCpu = (points: DataTable, centroids: DataTable, labels: Uint32Array) => {
    const kdTree = new KdTree(centroids);

    // construct a kdtree over the centroids so we can find the nearest quickly
    const point = new Float32Array(points.numColumns);
    const row: any = {};

    // assign each point to the nearest centroid
    for (let i = 0; i < points.numRows; ++i) {
        points.getRow(i, row);
        points.columns.forEach((c, i) => {
            point[i] = row[c.name];
        });

        const a = kdTree.findNearest(point);

        labels[i] = a.index;
    }
};

const kmeans = async (points: DataTable, k: number, iterations: number, device?: GraphicsDevice) => {
    // too few data points
    if (points.numRows < k) {
        return {
            centroids: points.clone(),
            // use a typed array here so downstream code can rely on
            // labels supporting subarray(), even in this early-return
            // path used for very small datasets.
            labels: new Uint32Array(points.numRows).map((_, i) => i)
        };
    }

    const row: any = {};

    // construct centroids data table and assign initial values
    const centroids = new DataTable(points.columns.map(c => new Column(c.name, new Float32Array(k))));
    if (points.numColumns === 1) {
        initializeCentroids1D(points, centroids);
    } else {
        initializeCentroids(points, centroids, row);
    }

    const gpuClustering = device && new GpuClustering(device, points, k);
    const labels = new Uint32Array(points.numRows);
    const counts = new Uint32Array(k);
    const sums = new Float64Array(k);

    let converged = false;
    let steps = 0;

    logger.debug(`running k-means clustering: dims=${points.numColumns} points=${points.numRows} clusters=${k} iterations=${iterations}`);

    const bar = logger.bar('k-means', iterations);
    try {
        while (!converged) {
            if (gpuClustering) {
                await gpuClustering.execute(centroids, labels);
            } else {
                clusterKdTreeCpu(points, centroids, labels);
            }

            counts.fill(0);
            for (let i = 0; i < labels.length; ++i) {
                counts[labels[i]]++;
            }
            for (let c = 0; c < points.numColumns; ++c) {
                const source = points.columns[c].data;
                const target = centroids.columns[c].data;
                sums.fill(0);
                // Ascending rows and f64 sums preserve each original centroid's reduction order.
                for (let i = 0; i < labels.length; ++i) {
                    sums[labels[i]] += source[i];
                }
                for (let i = 0; i < k; ++i) {
                    if (counts[i] > 0) {
                        target[i] = sums[i] / counts[i];
                    }
                }
            }
            for (let i = 0; i < centroids.numRows; ++i) {
                if (counts[i] === 0) {
                    // re-seed this centroid to a random point to avoid zero vector
                    const idx = Math.floor(Math.random() * points.numRows);
                    points.getRow(idx, row);
                    centroids.setRow(i, row);
                }
            }

            steps++;

            if (steps >= iterations) {
                converged = true;
            }

            bar.tick();
        }
    } catch (e) {
        gpuClustering?.destroy();
        throw e;
    }

    bar.end();
    gpuClustering?.destroy();

    return { centroids, labels };
};

export { kmeans };
