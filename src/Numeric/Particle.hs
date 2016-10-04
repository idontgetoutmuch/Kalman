-----------------------------------------------------------------------------
-- |
-- Module      :  Numeric.Particle
-- Copyright   :  (c) 2016 FP Complete Corporation
-- License     :  MIT (see LICENSE)
-- Maintainer  :  dominic@steinitz.org
--
-- =The Theory
-- The model for the particle filter is given by
--
-- \[
-- \begin{aligned}
-- \boldsymbol{x}_i &= \boldsymbol{a}_i(\boldsymbol{x}_{i-1}) + \boldsymbol{\psi}_{i-1} \\
-- \boldsymbol{y}_i &= \boldsymbol{h}_i(\boldsymbol{x}_i) + \boldsymbol{\upsilon}_i
-- \end{aligned}
-- \]
--
-- where
--
-- * \(\boldsymbol{a_i}\)\ is some non-linear vector-valued possibly
-- time-varying state update function.
--
-- * \(\boldsymbol{\psi}_{i}\) are independent normally
-- distributed random variables with mean 0 representing the fact that
-- the state update is noisy: \(\boldsymbol{\psi}_{i} \sim {\cal{N}}(0,Q_i)\).
--
-- * \(\boldsymbol{h}_i\)\ is some non-linear vector-valued possibly time-varying
-- function describing how we observe the hidden state process.
--
-- * \(\boldsymbol{\upsilon}_i\) are independent normally
-- distributed random variables with mean 0 represent the fact that
-- the observations are noisy: \(\boldsymbol{\upsilon}_{i} \sim {\cal{N}}(0,R_i)\).
--
-- Note that in most presentations of the Kalman filter (the
-- [wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
-- presentation being an exception), the state update function and the
-- observation function are taken to be constant over time as is the
-- noise for the state and the noise for
-- the observation. In symbols, \(\forall i \, \boldsymbol{a}_i = \boldsymbol{a}\)
-- for some \(\boldsymbol{a}\), \(\forall i \, \boldsymbol{h}_i = \boldsymbol{h}\)
-- for some \(\boldsymbol{h}\),
-- \(\forall i \, Q_i = Q\) for some \(Q\) and \(\forall i \, \boldsymbol{a}_i = R\)
-- for some \(R\).
--
-- We assume the whole process starts at 0 with our belief of the state (aka the
-- prior state) being given by
-- \(\boldsymbol{x}_0 \sim {\cal{N}}(\boldsymbol{\mu}_0, \Sigma_0)\)
--
-- The Kalman filtering process is a recursive procedure as follows:
--
-- (1) Make a prediction of the current system state, given our previous
-- estimation of the system state.
--
-- (2) Update our prediction, given a newly acquired measurement.
--
-- The prediction and update steps depend on our system and measurement
-- models, as well as our current estimate of the system state.
--
-- = An Extended Example
--
-- The equation of motion for a pendulum of unit length subject to
-- [Gaussian white
-- noise](https://en.wikipedia.org/wiki/White_noise#Mathematical_definitions)
-- is
--
-- \[
-- \frac{\mathrm{d}^2\alpha}{\mathrm{d}t^2} = -g\sin\alpha + w(t)
-- \]
--
-- We can discretize this via the usual [Euler method](https://en.wikipedia.org/wiki/Euler_method)
--
-- \[
-- \begin{bmatrix}
-- x_{1,i} \\
-- x_{2,i}
-- \end{bmatrix}
-- =
-- \begin{bmatrix}
-- x_{1,i-1} + x_{2,i-1}\Delta t \\
-- x_{2,i-1} - g\sin x_{1,i-1}\Delta t
-- \end{bmatrix}
-- +
-- \mathbf{q}_{i-1}
-- \]
--
-- where \(q_i \sim {\mathcal{N}}(0,Q)\) and
--
-- \[
-- Q
-- =
-- \begin{bmatrix}
-- \frac{q^c \Delta t^3}{3} & \frac{q^c \Delta t^2}{2} \\
-- \frac{q^c \Delta t^2}{2} & {q^c \Delta t}
-- \end{bmatrix}
-- \]
--
-- Assume that we can only measure the horizontal position of the
-- pendulum and further that this measurement is subject to error so that
--
-- \[
-- y_i = \sin x_i + r_i
-- \]
--
-- where \(r_i \sim {\mathcal{N}}(0,R)\).
--
-- First let's set the time step and the acceleration caused by earth's gravity.
--
-- > {-# LANGUAGE DataKinds #-}
-- >
-- > import Numeric.Kalman
-- >
-- > deltaT, g :: Double
-- > deltaT = 0.01
-- > g  = 9.81
-- > bigQ :: Sym 2
-- > bigQ = sym $ matrix bigQl
-- >
-- > qc :: Double
-- > qc = 0.01
-- >
-- > bigQl :: [Double]
-- > bigQl = [ qc * deltaT^3 / 3, qc * deltaT^2 / 2,
-- >           qc * deltaT^2 / 2,       qc * deltaT
-- >         ]
-- >
-- > bigR :: Sym 1
-- > bigR  = sym $ matrix [0.1]
-- > stateUpdate :: R 2 -> R 2
-- > stateUpdate u =  vector [x1 + x2 * deltaT, x2 - g * (sin x1) * deltaT]
-- >   where
-- >     (x1, w) = headTail u
-- >     (x2, _) = headTail w
-- >
-- > observe :: R 2 -> R 1
-- > observe a = vector [sin x] where x = fst $ headTail a
-- > linearizedObserve :: R 2 -> L 1 2
-- > linearizedObserve a = matrix [cos x, 0.0] where x = fst $ headTail a
-- >
-- > linearizedStateUpdate :: R 2 -> Sq 2
-- > linearizedStateUpdate u = matrix [1.0,                    deltaT,
-- >                                   -g * (cos x1) * deltaT,    1.0]
-- >   where
-- >     (x1, _) = headTail u
--
-- Now we can create extended and unscented filters which consume a
-- single observation.
--
-- > singleEKF = runEKF (const observe) (const linearizedObserve) (const bigR)
-- >              (const stateUpdate) (const linearizedStateUpdate) (const bigQ)
-- >              undefined
--
-- > singleUKF = runUKF (const observe) (const bigR) (const stateUpdate) (const bigQ)
-- >              undefined
--
-- We start off not too far from the actual value.
--
-- > initialDist =  (vector [1.6, 0.0],
-- >                 sym $ matrix [0.1, 0.0,
-- >                               0.0, 0.1])
--
-- Using some data generated using code made available with Simo
-- Särkkä's
-- [book](http://www.cambridge.org/gb/academic/subjects/statistics-probability/applied-probability-and-stochastic-networks/bayesian-filtering-and-smoothing),
-- we can track the pendulum using the extended Kalman filter.
--
-- > multiEKF obs = scanl singleEKF initialDist (map (vector . pure) obs)
--
-- And then plot the results.
--
-- <<diagrams/src_Numeric_Particle_diagEP.svg#diagram=diagEP&height=600&width=500>>
--
-- And also track it using the unscented Kalman filter.
--
-- > multiUKF obs = scanl singleUKF initialDist (map (vector . pure) obs)
--
-- And also plot the results
--
-- <<diagrams/src_Numeric_Particle_diagUP.svg#diagram=diagUP&height=600&width=500>>
--
-- === Code for Plotting
--
-- The full code for plotting the results:
--
-- > import qualified Graphics.Rendering.Chart as C
-- > import Graphics.Rendering.Chart.Backend.Diagrams
-- > import Data.Colour
-- > import Data.Colour.Names
-- > import Data.Default.Class
-- > import Control.Lens
-- >
-- > import Data.Csv
-- > import System.IO hiding ( hGetContents )
-- > import Data.ByteString.Lazy ( hGetContents )
-- > import qualified Data.Vector as V
-- >
-- > import Numeric.LinearAlgebra.Static
-- >
-- > chartEstimated :: String ->
-- >               [(Double, Double)] ->
-- >               [(Double, Double)] ->
-- >               [(Double, Double)] ->
-- >               C.Renderable ()
-- > chartEstimated title acts obs ests = C.toRenderable layout
-- >   where
-- >
-- >     actuals = C.plot_lines_values .~ [acts]
-- >             $ C.plot_lines_style  . C.line_color .~ opaque red
-- >             $ C.plot_lines_title .~ "Actual Trajectory"
-- >             $ C.plot_lines_style  . C.line_width .~ 1.0
-- >             $ def
-- >
-- >     measurements = C.plot_points_values .~ obs
-- >                  $ C.plot_points_style  . C.point_color .~ opaque blue
-- >                  $ C.plot_points_title .~ "Measurements"
-- >                  $ def
-- >
-- >     estimas = C.plot_lines_values .~ [ests]
-- >             $ C.plot_lines_style  . C.line_color .~ opaque black
-- >             $ C.plot_lines_title .~ "Inferred Trajectory"
-- >             $ C.plot_lines_style  . C.line_width .~ 1.0
-- >             $ def
-- >
-- >     layout = C.layout_title .~ title
-- >            $ C.layout_plots .~ [C.toPlot actuals, C.toPlot measurements, C.toPlot estimas]
-- >            $ C.layout_y_axis . C.laxis_title .~ "Angle / Horizontal Displacement"
-- >            $ C.layout_y_axis . C.laxis_override .~ C.axisGridHide
-- >            $ C.layout_x_axis . C.laxis_title .~ "Time"
-- >            $ C.layout_x_axis . C.laxis_override .~ C.axisGridHide
-- >            $ def
-- >
-- > diagEP = do
-- >   h <- openFile "matlabRNGs.csv" ReadMode
-- >   cs <- hGetContents h
-- >   let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
-- >   case df of
-- >     Left _ -> error "Whatever"
-- >     Right generatedSamples -> do
-- >       let xs = take 500 (multiEKF $ V.toList $ V.map fst generatedSamples)
-- >       let mus = map (fst . headTail . fst) xs
-- >       let obs = V.toList $ V.map fst generatedSamples
-- >       let acts = V.toList $ V.map snd generatedSamples
-- >       denv <- defaultEnv C.vectorAlignmentFns 600 500
-- >       let charte = chartEstimated "Extended Kalman Filter"
-- >                                   (zip [0,1..] acts)
-- >                                   (zip [0,1..] obs)
-- >                                   (zip [0,1..] mus)
-- >       return $ fst $ runBackend denv (C.render charte (600, 500))
-- >
-- > diagUP = do
-- >   h <- openFile "matlabRNGs.csv" ReadMode
-- >   cs <- hGetContents h
-- >   let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
-- >   case df of
-- >     Left _ -> error "Whatever"
-- >     Right generatedSamples -> do
-- >       let ys = take 500 (multiUKF $ V.toList $ V.map fst generatedSamples)
-- >       let nus = map (fst . headTail . fst) ys
-- >       let obs = V.toList $ V.map fst generatedSamples
-- >       let acts = V.toList $ V.map snd generatedSamples
-- >       denv <- defaultEnv C.vectorAlignmentFns 600 500
-- >       let charte = chartEstimated "Unscented Kalman Filter"
-- >                                   (zip [0,1..] acts)
-- >                                   (zip [0,1..] obs)
-- >                                   (zip [0,1..] nus)
-- >       return $ fst $ runBackend denv (C.render charte (600, 500))
--
-----------------------------------------------------------------------------

{-# OPTIONS_GHC -Wall                     #-}

{-# LANGUAGE BangPatterns                 #-}

module Numeric.Particle (
    runPF
  , oneSmoothingPath
  , oneSmoothingStep
  , scanMapM
  , Particles ) where

import           Data.Random hiding ( StdNormal, Normal )
import           Control.Monad
import qualified Data.Vector as V
import           Data.Vector ( Vector )
import           Data.Bits ( shiftR )

type Particles a = Vector a
type Path a      = Vector a

runPF
  :: MonadRandom m
  => (Particles a -> m (Particles a)) -- ^ System evolution at a point
  -> (Particles a -> Particles b)     -- ^ Measurement operator at a point
  -> (b -> b -> Double)               -- ^ Observation probability density function
  -> Particles a                      -- ^ Current estimate
  -> b                                -- ^ New measurement
  -> m (Particles a)                  -- ^ New estimate
runPF stateUpdate obsUpdate weight statePrevs obs = do
  stateNews <- stateUpdate statePrevs
  let obsNews = obsUpdate stateNews
  let weights       = V.map (weight obs) obsNews
      cumSumWeights = V.tail $ V.scanl (+) 0 weights
      totWeight     = V.last cumSumWeights
      nParticles    = V.length statePrevs
  vs <- V.replicateM nParticles (sample $ uniform 0.0 totWeight)
  let js = indices cumSumWeights vs
      stateTildes = V.map (stateNews V.!) js
  return stateTildes

oneSmoothingPath
  :: MonadRandom m
  => (Particles a -> m (Particles a)) -- ^ System evolution at a point
  -> (a -> a -> Double) -- ^ State probability density function
  -> Int
  -> (Vector (Particles a))
  -> m (Path a)
oneSmoothingPath stateUpdate weight nParticles filterEstss = do
  let ys = filterEstss
  ix <- sample $ uniform 0 (nParticles - 1)
  let xn = (V.head ys) V.! ix
  scanMapM (oneSmoothingStep stateUpdate weight) return xn (V.tail ys)

oneSmoothingStep :: MonadRandom m =>
         (Particles a -> m (Particles a)) ->
         (a -> a -> Double) ->
         a ->
         Particles a ->
         m a
oneSmoothingStep stateUpdate
                 stateDensity
                 smoothingSampleAtiPlus1
                 filterSamplesAti = do it
  where
    it = do
      mus <- stateUpdate filterSamplesAti
      let weights =  V.map (stateDensity smoothingSampleAtiPlus1) mus
          cumSumWeights = V.tail $ V.scanl (+) 0 weights
          totWeight     = V.last cumSumWeights
      v <- sample $ uniform 0.0 totWeight
      let ix = binarySearch cumSumWeights v
          xnNew = filterSamplesAti V.! ix
      return $ xnNew

indices :: Vector Double -> Vector Double -> Vector Int
indices bs xs = V.map (binarySearch bs) xs

binarySearch :: (Ord a) =>
                Vector a -> a -> Int
binarySearch vec x = loop 0 (V.length vec - 1)
  where
    loop !l !u
      | u <= l    = l
      | otherwise = let e = vec V.! k in if x <= e then loop l k else loop (k+1) u
      where k = l + (u - l) `shiftR` 1

scanMapM :: Monad m => (s -> a -> m s) -> (s -> m b) -> s -> Vector a -> m (Vector b)
scanMapM f g !s0 !xs
  | V.null xs = do
    r <- g s0
    return $ V.singleton r
  | otherwise = do
    s <- f s0 (V.head xs)
    r <- g s0
    liftM (r `V.cons`) (scanMapM f g s (V.tail xs))
