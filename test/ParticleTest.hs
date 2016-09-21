{-# OPTIONS_GHC -Wall                     #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing  #-}
{-# OPTIONS_GHC -fno-warn-type-defaults   #-}
{-# OPTIONS_GHC -fno-warn-unused-do-bind  #-}
{-# OPTIONS_GHC -fno-warn-missing-methods #-}
{-# OPTIONS_GHC -fno-warn-orphans         #-}

{-# LANGUAGE MultiParamTypeClasses        #-}
{-# LANGUAGE TypeFamilies                 #-}
{-# LANGUAGE ScopedTypeVariables          #-}
{-# LANGUAGE ExplicitForAll               #-}
{-# LANGUAGE DataKinds                    #-}


{-# LANGUAGE FlexibleInstances            #-}
{-# LANGUAGE MultiParamTypeClasses        #-}
{-# LANGUAGE FlexibleContexts             #-}
{-# LANGUAGE TypeFamilies                 #-}
{-# LANGUAGE BangPatterns                 #-}
{-# LANGUAGE GeneralizedNewtypeDeriving   #-}
{-# LANGUAGE TemplateHaskell              #-}
{-# LANGUAGE DataKinds                    #-}
{-# LANGUAGE DeriveGeneric                #-}

module Particle where

import           Data.Random hiding ( StdNormal, Normal )
import           Data.Random.Source.PureMT ( pureMT )
import           Control.Monad.State ( evalState, replicateM )
import qualified Control.Monad.Loops as ML
import           Numeric.LinearAlgebra.Static
                 ( R, vector, Sym,
                   headTail, matrix, sym,
                   diag
                 )
import           GHC.TypeLits ( KnownNat )
import           Data.Random.Distribution.Static.MultivariateNormal ( Normal(..) )
import qualified Data.Vector as V
import           Data.Vector ( Vector )
import           Data.List ( transpose )
import           Control.Parallel.Strategies
import           GHC.Generics (Generic)

import           Numeric.Particle

import           Debug.Trace

nParticles :: Int
nParticles = 100 -- 500

nTimesteps :: Int
nTimesteps = 10 -- 7

data SystemState a = SystemState { x1  :: a, x2  :: a }
  deriving (Show, Generic)

instance NFData a => NFData (SystemState a)

newtype SystemObs a = SystemObs { y1  :: a }
  deriving Show

(.+), (.*), (.-) :: (Num a) => V.Vector a -> V.Vector a -> V.Vector a
(.+) = V.zipWith (+)
(.*) = V.zipWith (*)
(.-) = V.zipWith (-)

deltaT, g :: Double
deltaT = 0.01
g  = 9.81

type PendulumState = R 2
type PendulumObs = R 1

pendulumSample :: MonadRandom m =>
                  Sym 2 ->
                  Sym 1 ->
                  PendulumState ->
                  m (Maybe ((PendulumState, PendulumObs), PendulumState))
pendulumSample bigQ bigR xPrev = do
  let x1Prev = fst $ headTail xPrev
      x2Prev = fst $ headTail $ snd $ headTail xPrev
  eta <- sample $ rvar (Normal 0.0 bigQ)
  let x1= x1Prev + x2Prev * deltaT
      x2 = x2Prev - g * (sin x1Prev) * deltaT
      xNew = vector [x1, x2] + eta
      x1New = fst $ headTail xNew
  epsilon <-  sample $ rvar (Normal 0.0 bigR)
  let yNew = vector [sin x1New] + epsilon
  return $ Just ((xNew, yNew), xNew)

stateUpdate :: Particles (SystemState Double) ->
                Particles (SystemState Double)
stateUpdate xPrevs = V.zipWith SystemState x1s x2s
  where
    ix = V.length xPrevs

    x1Prevs = V.map x1 xPrevs
    x2Prevs = V.map x2 xPrevs

    deltaTs = V.replicate ix deltaT
    gs = V.replicate ix g
    x1s = x1Prevs .+ (x2Prevs .* deltaTs)
    x2s = x2Prevs .- (gs .* (V.map sin x1Prevs) .* deltaTs)

stateUpdateNoisy :: MonadRandom m =>
                    Sym 2 ->
                    Particles (SystemState Double) ->
                    m (Particles (SystemState Double))
stateUpdateNoisy bigQ xPrevs = do
  let xs = stateUpdate xPrevs

      x1s = V.map x1 xs
      x2s = V.map x2 xs

  let ix = V.length xPrevs
  etas <- replicateM ix $ sample $ rvar (Normal 0.0 bigQ)

  let eta1s, eta2s :: V.Vector Double
      eta1s = V.fromList $ map (fst . headTail) etas
      eta2s = V.fromList $ map (fst . headTail . snd . headTail) etas

  return (V.zipWith SystemState (x1s .+ eta1s) (x2s .+ eta2s))

obsUpdate :: Particles (SystemState Double) ->
             Particles (SystemObs Double)
obsUpdate xs = V.map (SystemObs . sin . x1) xs

weight :: forall a n . KnownNat n =>
          (a -> R n) ->
          Sym n ->
          a -> a -> Double
weight f bigR obs obsNew = pdf (Normal (f obsNew) bigR) (f obs)

bigP :: Sym 2
bigP = sym $ diag 0.1

initParticles :: MonadRandom m =>
                 m (Particles (SystemState Double))
initParticles = V.replicateM nParticles $ do
  r <- sample $ rvar (Normal m0' bigP)
  let x1 = fst $ headTail r
      x2 = fst $ headTail $ snd $ headTail r
  return $ SystemState { x1 = x1, x2 = x2}

runFilter :: Int -> Vector (Particles (SystemState Double))
runFilter nTimeSteps = evalState action (pureMT 19)
  where
    action = do
      xs <- initParticles
      scanMapM
        (runPF (stateUpdateNoisy bigQ') obsUpdate (weight f bigR'))
        return
        xs
        (V.fromList $ map (SystemObs . fst . headTail . snd)
                          (take nTimeSteps pendulumSamples'))



f :: SystemObs Double -> R 1
f = vector . pure . y1

h :: SystemState Double -> R 2
h u = vector [x1 u , x2 u]

testFiltering :: Int -> Vector Double
testFiltering nTimeSteps = V.map ((/ (fromIntegral nParticles)). V.sum . V.map x1)
                                 (runFilter nTimeSteps)

bigQ' :: Sym 2
bigQ' = sym $ matrix bigQl'

qc1' :: Double
qc1' = 0.01

bigQl' :: [Double]
bigQl' = [ qc1' * deltaT^3 / 3, qc1' * deltaT^2 / 2,
           qc1' * deltaT^2 / 2,       qc1' * deltaT
         ]

bigR' :: Sym 1
bigR'  = sym $ matrix [0.1]

m0' :: PendulumState
m0' = vector [1.6, 0]

pendulumSamples' :: [(PendulumState, PendulumObs)]
pendulumSamples' = evalState (ML.unfoldrM (pendulumSample bigQ' bigR') m0') (pureMT 17)

filterEstss :: V.Vector (Particles (SystemState Double))
filterEstss = V.reverse $ runFilter nTimesteps

testSmoothing' :: Int -> [Double]
testSmoothing' n = V.toList $ evalState action (pureMT 23)
  where
    action = do
      xss <- V.replicateM n $ oneSmoothingPath (stateUpdateNoisy bigQ') (weight h bigQ') nParticles filterEstss
      let yss = V.fromList $ map V.fromList $
                transpose $
                V.toList $ V.map (V.toList) $
                xss
      trace (show xss) $
        return $ V.map (/ (fromIntegral n)) $ V.map V.sum $ V.map (V.map x1) yss
