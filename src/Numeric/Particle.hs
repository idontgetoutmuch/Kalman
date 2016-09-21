{-# OPTIONS_GHC -Wall                     #-}

{-# LANGUAGE BangPatterns                 #-}

module Numeric.Particle where

import           Data.Random hiding ( StdNormal, Normal )
import           Control.Monad.Writer ( tell, WriterT, lift,
                                        runWriterT
                                      )
import           Control.Monad
import qualified Data.Vector as V
import           Data.Vector ( Vector )
import           Data.Bits ( shiftR )

type Particles a = Vector a
type Path a      = Vector a

runPF
  :: MonadRandom m
  => (Particles a -> m (Vector a)) -- ^ System evolution at a point
  -> (Particles a -> Particles b)    -- ^ Measurement operator at a point
  -> (b -> b -> Double)              -- ^ Observation probability density function
  -> Particles a                     -- ^ Current estimate
  -> b                               -- ^ New measurement
  -> m (Particles a)                 -- ^ New estimate
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

oneSmoothingPath'' :: MonadRandom m  =>
                      Int ->
                      (Int -> Vector (Particles a)) ->
                      (a -> Particles a -> m a) ->
                      Int ->
                      m (Path a)
oneSmoothingPath'' nParticles filterEstss ss nTimeSteps = do
  let ys = filterEstss nTimeSteps
  ix <- sample $ uniform 0 (nParticles - 1)
  let xn = (V.head ys) V.! ix
  scanMapM ss return xn (V.tail ys)

oneSmoothingStep' :: MonadRandom m =>
         (Particles a -> Particles a) ->
         (a -> a -> Double) ->
         a ->
         Particles a ->
         m a
oneSmoothingStep' stateUpdate
                 stateDensity
                 smoothingSampleAtiPlus1
                 filterSamplesAti = do it
  where
    it = do
      let mus = stateUpdate filterSamplesAti
          weights =  V.map (stateDensity smoothingSampleAtiPlus1) mus
          cumSumWeights = V.tail $ V.scanl (+) 0 weights
          totWeight     = V.last cumSumWeights
      v <- sample $ uniform 0.0 totWeight
      let ix = binarySearch cumSumWeights v
          xnNew = filterSamplesAti V.! ix
      return $ xnNew

oneSmoothingStep :: MonadRandom m =>
         (Particles a -> Particles a) ->
         (a -> a -> Double) ->
         a ->
         Particles a ->
         WriterT (Particles a) m a
oneSmoothingStep stateUpdate
                 stateDensity
                 smoothingSampleAtiPlus1
                 filterSamplesAti = do it
  where
    it = do
      let mus = stateUpdate filterSamplesAti
          weights =  V.map (stateDensity smoothingSampleAtiPlus1) mus
          cumSumWeights = V.tail $ V.scanl (+) 0 weights
          totWeight     = V.last cumSumWeights
      v <- lift $ sample $ uniform 0.0 totWeight
      let ix = binarySearch cumSumWeights v
          xnNew = filterSamplesAti V.! ix
      tell $ V.singleton xnNew
      return $ xnNew

oneSmoothingPath :: MonadRandom m =>
                    Int ->
                    (Int -> Vector (Particles a)) ->
                    (a -> Particles a -> WriterT (Particles a) m a) ->
                    Int -> m (a, Particles a)
oneSmoothingPath nParticles filterEstss ss nTimeSteps = do
  let ys = filterEstss nTimeSteps
  ix <- sample $ uniform 0 (nParticles - 1)
  let xn = (V.head ys) V.! ix
  runWriterT $ V.foldM ss xn (V.tail ys)

oneSmoothingPath' :: MonadRandom m =>
                     Int ->
                     (Int -> Vector (Particles a)) ->
                     (a -> Particles a -> WriterT (Particles a) m a) ->
                     Int -> WriterT (Particles a) m a
oneSmoothingPath' nParticles filterEstss ss nTimeSteps = do
  let ys = filterEstss nTimeSteps
  ix <- lift $ sample $ uniform 0 (nParticles - 1)
  let xn = (V.head ys) V.! ix
  V.foldM ss xn (V.tail ys)

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
