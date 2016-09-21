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
