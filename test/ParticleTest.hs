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

module Particle ( main ) where

import           Data.Random hiding ( StdNormal, Normal )
import           Data.Random.Source.PureMT ( pureMT )
import           Control.Monad.State ( evalState, replicateM )
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

import qualified Graphics.Rendering.Chart as C
import Graphics.Rendering.Chart.Backend.Diagrams
import Data.Colour
import Data.Colour.Names
import Data.Default.Class
import Control.Lens

import Diagrams.Backend.Cairo.CmdLine
import Diagrams.Prelude hiding ( render, Renderable, trace, Vector, sample )
import Diagrams.Backend.CmdLine

import Data.Csv
import System.IO hiding ( hGetContents )
import Data.ByteString.Lazy ( hGetContents )

nParticles :: Int
nParticles = 1000 -- 500

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
  r <- sample $ rvar (Normal m0 bigP)
  let x1 = fst $ headTail r
      x2 = fst $ headTail $ snd $ headTail r
  return $ SystemState { x1 = x1, x2 = x2}

runFilter :: Particles (SystemObs Double) -> Vector (Particles (SystemState Double))
runFilter pendulumSamples = evalState action (pureMT 19)
  where
    action = do
      xs <- initParticles
      scanMapM
        (runPF (stateUpdateNoisy bigQ) obsUpdate (weight f bigR))
        return
        xs
        pendulumSamples

f :: SystemObs Double -> R 1
f = vector . pure . y1

h :: SystemState Double -> R 2
h u = vector [x1 u , x2 u]

bigQ :: Sym 2
bigQ = sym $ matrix bigQl

qc1 :: Double
qc1 = 0.01

bigQl :: [Double]
bigQl = [ qc1 * deltaT^3 / 3, qc1 * deltaT^2 / 2,
          qc1 * deltaT^2 / 2,       qc1 * deltaT
         ]

bigR :: Sym 1
bigR  = sym $ matrix [0.1]

m0 :: PendulumState
m0 = vector [1.6, 0]

testSmoothing :: Particles (SystemObs Double) -> Int -> [Double]
testSmoothing ss n = V.toList $ evalState action (pureMT 23)
  where
    action = do
      xss <- V.replicateM n $ oneSmoothingPath (stateUpdateNoisy bigQ) (weight h bigQ) nParticles (runFilter ss)
      let yss = V.fromList $ map V.fromList $
                transpose $
                V.toList $ V.map (V.toList) $
                xss
      return $ V.map (/ (fromIntegral n)) $ V.map V.sum $ V.map (V.map x1) yss

chartEstimated :: String ->
              [(Double, Double)] ->
              [(Double, Double)] ->
              [(Double, Double)] ->
              C.Renderable ()
chartEstimated title acts obs ests = C.toRenderable layout
  where

    actuals = C.plot_lines_values .~ [acts]
            $ C.plot_lines_style  . C.line_color .~ opaque red
            $ C.plot_lines_title .~ "Actual Trajectory"
            $ C.plot_lines_style  . C.line_width .~ 1.0
            $ def

    measurements = C.plot_points_values .~ obs
                 $ C.plot_points_style  . C.point_color .~ opaque blue
                 $ C.plot_points_title .~ "Measurements"
                 $ def

    estimas = C.plot_lines_values .~ [ests]
            $ C.plot_lines_style  . C.line_color .~ opaque black
            $ C.plot_lines_title .~ "Inferred Trajectory"
            $ C.plot_lines_style  . C.line_width .~ 1.0
            $ def

    layout = C.layout_title .~ title
           $ C.layout_plots .~ [C.toPlot actuals, C.toPlot measurements, C.toPlot estimas]
           $ C.layout_y_axis . C.laxis_title .~ "Angle / Horizontal Displacement"
           $ C.layout_y_axis . C.laxis_override .~ C.axisGridHide
           $ C.layout_x_axis . C.laxis_title .~ "Time"
           $ C.layout_x_axis . C.laxis_override .~ C.axisGridHide
           $ def

nObs :: Int
nObs = 200

diagU :: IO (Diagram Cairo)
diagU = do
  h <- openFile "matlabRNGs.csv" ReadMode
  cs <- hGetContents h
  let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
  case df of
    Left _ -> error "Whatever"
    Right generatedSamples -> do
      let preObs = V.take nObs $ V.map fst generatedSamples
      let obs = V.toList preObs
      let acts = V.toList $ V.take nObs $ V.map snd generatedSamples
      let nus = take nObs (testSmoothing (V.map SystemObs preObs) 50)
      denv <- defaultEnv C.vectorAlignmentFns 600 500
      let charte = chartEstimated "Particle Smoother"
                                  (zip [0,1..] acts)
                                  (zip [0,1..] obs)
                                  (zip [0,1..] nus)
      return $ fst $ runBackend denv (C.render charte (600, 500))

displayHeader :: FilePath -> Diagram B -> IO ()
displayHeader fn =
  mainRender ( DiagramOpts (Just 900) (Just 700) fn
             , DiagramLoopOpts False Nothing 0
             )

main :: IO ()
main = do
  du <- diagU
  displayHeader "diagrams/Smooth3.png" du
