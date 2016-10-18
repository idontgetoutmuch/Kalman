-----------------------------------------------------------------------------
-- |
-- Module      :  Numeric.Particle
-- Copyright   :  (c) 2016 FP Complete Corporation
-- License     :  MIT (see LICENSE)
-- Maintainer  :  dominic@steinitz.org
--
-- =The Theory
--
-- The particle filter, `runPF`, in this library is applicable to the
-- state space model given by
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
-- Clearly this could be generalised further; anyone wishing for such
-- a generalisation is encouraged to contribute to the library.
--
-- The smoother, `oneSmoothingPath` implements [Forward filtering /
-- backward
-- smoothing](https://en.wikipedia.org/wiki/Particle_filter#Backward_particle_smoothers)
-- and returns just one path from the particle filter; in most cases,
-- this will need to be run many times to provide good estimates of
-- the past. Note that `oneSmoothingPath` uses /all/ observation up
-- until the current time. This could be generalised to select only a
-- window of observations up to the current time. Again, contributions
-- to implement this generalisation are welcomed.
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
-- > import Numeric.Particle
-- >
-- > deltaT, g :: Double
-- > deltaT = 0.01
-- > g  = 9.81
-- >
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
-- >
-- > data SystemState a = SystemState { x1  :: a, x2  :: a }
-- >   deriving Show
-- >
-- > newtype SystemObs a = SystemObs { y1  :: a }
-- >   deriving Show
-- >
--
-- Bar
--
-- > {-# LANGUAGE DataKinds #-}
-- >
-- > m0 :: PendulumState
-- > m0 = vector [1.6, 0]
-- >
-- > bigP :: Sym 2
-- > bigP = sym $ diag 0.1
-- >
-- > initParticles :: R.MonadRandom m =>
-- >                  m (Particles (SystemState Double))
-- > initParticles = V.replicateM nParticles $ do
-- >   r <- R.sample $ R.rvar (Normal m0 bigP)
-- >   let x1 = fst $ headTail r
-- >       x2 = fst $ headTail $ snd $ headTail r
-- >   return $ SystemState { x1 = x1, x2 = x2}
-- >
-- > nObs :: Int
-- > nObs = 35
-- >
-- > nParticles :: Int
-- > nParticles = 20
-- >
-- > (.+), (.*), (.-) :: (Num a) => V.Vector a -> V.Vector a -> V.Vector a
-- > (.+) = V.zipWith (+)
-- > (.*) = V.zipWith (*)
-- > (.-) = V.zipWith (-)
-- >
-- > stateUpdateP :: Particles (SystemState Double) ->
-- >                 Particles (SystemState Double)
-- > stateUpdateP xPrevs = V.zipWith SystemState x1s x2s
-- >   where
-- >     ix = V.length xPrevs
-- >
-- >     x1Prevs = V.map x1 xPrevs
-- >     x2Prevs = V.map x2 xPrevs
-- >
-- >     deltaTs = V.replicate ix deltaT
-- >     gs = V.replicate ix g
-- >     x1s = x1Prevs .+ (x2Prevs .* deltaTs)
-- >     x2s = x2Prevs .- (gs .* (V.map sin x1Prevs) .* deltaTs)
--
-- We start off not too far from the actual value.
--
-- Using some data generated using code made available with Simo
-- Särkkä's
-- [book](http://www.cambridge.org/gb/academic/subjects/statistics-probability/applied-probability-and-stochastic-networks/bayesian-filtering-and-smoothing),
-- we can track the pendulum using the extended Kalman filter.
--
-- AND also plot the results
--
-- <<diagrams/src_Numeric_Particle_diagV.svg#diagram=diagV&height=600&width=500>>
--
-- === Code for Plotting
--
-- The full code for plotting the results:
--
-- > {-# LANGUAGE ExplicitForAll #-}
-- > {-# LANGUAGE TypeOperators  #-}
-- > {-# LANGUAGE DataKinds      #-}
-- >
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
-- > import           Data.Random.Distribution.Static.MultivariateNormal ( Normal(..) )
-- > import qualified Data.Random as R
-- > import           Data.Random.Source.PureMT ( pureMT )
-- > import           Control.Monad.State ( evalState, replicateM )
-- >
-- > import Data.List ( transpose )
-- >
-- > import GHC.TypeLits ( KnownNat )
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
-- > stateUpdateNoisy :: R.MonadRandom m =>
-- >                     Sym 2 ->
-- >                     Particles (SystemState Double) ->
-- >                     m (Particles (SystemState Double))
-- > stateUpdateNoisy bigQ xPrevs = do
-- >   let xs = stateUpdateP xPrevs
-- >
-- >       x1s = V.map x1 xs
-- >       x2s = V.map x2 xs
-- >
-- >   let ix = V.length xPrevs
-- >   etas <- replicateM ix $ R.sample $ R.rvar (Normal 0.0 bigQ)
-- >
-- >   let eta1s, eta2s :: V.Vector Double
-- >       eta1s = V.fromList $ map (fst . headTail) etas
-- >       eta2s = V.fromList $ map (fst . headTail . snd . headTail) etas
-- >
-- >   return (V.zipWith SystemState (x1s .+ eta1s) (x2s .+ eta2s))
-- >
-- > obsUpdate :: Particles (SystemState Double) ->
-- >              Particles (SystemObs Double)
-- > obsUpdate xs = V.map (SystemObs . sin . x1) xs
-- >
-- > weight :: forall a n . KnownNat n =>
-- >           (a -> R n) ->
-- >           Sym n ->
-- >           a -> a -> Double
-- > weight f bigR obs obsNew = R.pdf (Normal (f obsNew) bigR) (f obs)
-- >
-- > runFilter :: Particles (SystemObs Double) -> V.Vector (Particles (SystemState Double))
-- > runFilter pendulumSamples = evalState action (pureMT 19)
-- >   where
-- >     action = do
-- >       xs <- initParticles
-- >       scanMapM
-- >         (runPF (stateUpdateNoisy bigQ) obsUpdate (weight f bigR))
-- >         return
-- >         xs
-- >         pendulumSamples
-- >
-- > testSmoothing :: Particles (SystemObs Double) -> Int -> [Double]
-- > testSmoothing ss n = V.toList $ evalState action (pureMT 23)
-- >   where
-- >     action = do
-- >       xss <- V.replicateM n $ oneSmoothingPath (stateUpdateNoisy bigQ) (weight h bigQ) nParticles (runFilter ss)
-- >       let yss = V.fromList $ map V.fromList $
-- >                 transpose $
-- >                 V.toList $ V.map (V.toList) $
-- >                 xss
-- >       return $ V.map (/ (fromIntegral n)) $ V.map V.sum $ V.map (V.map x1) yss
-- >
-- > type PendulumState = R 2
-- >
-- > f :: SystemObs Double -> R 1
-- > f = vector . pure . y1
-- >
-- > h :: SystemState Double -> R 2
-- > h u = vector [x1 u , x2 u]
-- >
-- > diagV = do
-- >   h <- openFile "matlabRNGs.csv" ReadMode
-- >   cs <- hGetContents h
-- >   let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
-- >   case df of
-- >     Left _ -> error "Whatever"
-- >     Right generatedSamples -> do
-- >       let preObs = V.take nObs $ V.map fst generatedSamples
-- >       let obs = V.toList preObs
-- >       let acts = V.toList $ V.take nObs $ V.map snd generatedSamples
-- >       let nus = take nObs (testSmoothing (V.map SystemObs preObs) 50)
-- >       denv <- defaultEnv C.vectorAlignmentFns 600 500
-- >       let charte = chartEstimated "Particle Smoother"
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
  , Particles
  , Path
  ) where

import           Data.Random hiding ( StdNormal, Normal )
import           Control.Monad
import qualified Data.Vector as V
import           Data.Vector ( Vector )
import           Data.Bits ( shiftR )

type Particles a = Vector a -- ^ As an aid for the reader
type Path a      = Vector a -- ^ As an aid for the reader

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
