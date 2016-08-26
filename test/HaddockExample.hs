{-# OPTIONS_GHC -Wall                   #-}
{-# OPTIONS_GHC -fno-warn-type-defaults #-}

{-# LANGUAGE DataKinds #-}

module HaddockExample where

import qualified Control.Monad.Loops as ML
import           Numeric.LinearAlgebra.Static
                 ( R, vector, Sym,
                   headTail, matrix, sym,
                   diag
                 )
import           Data.Random hiding ( StdNormal, Normal )
import           Data.Random.Source.PureMT ( pureMT )
import           Control.Monad.State ( evalState, replicateM )
import           MultivariateNormal ( MultivariateNormal(..) )
import           Data.Random.Distribution.MultiNormal
import           Numeric.Kalman
import           GHC.TypeLits ( KnownNat )
import           Numeric.LinearAlgebra.Static

import Graphics.Rendering.Chart
import Graphics.Rendering.Chart.Backend.Diagrams
import Diagrams.Backend.Cairo.CmdLine
import Diagrams.Prelude hiding ( render, Renderable, sample )
import Diagrams.Backend.CmdLine

import System.IO.Unsafe

deltaT, g :: Double
deltaT = 0.01
g  = 9.81

pendulumSample :: MonadRandom m =>
                  Sym 2 ->
                  Sym 1 ->
                  R 2 ->
                  m (Maybe ((R 2, R 1), R 2))
pendulumSample bigQ bigR xPrev = do
  let x1Prev = fst $ headTail xPrev
      x2Prev = fst $ headTail $ snd $ headTail xPrev
  eta <- sample $ rvar (MultivariateNormal 0.0 bigQ)
  let x1= x1Prev + x2Prev * deltaT
      x2 = x2Prev - g * (sin x1Prev) * deltaT
      xNew = vector [x1, x2] + eta
      x1New = fst $ headTail xNew
  epsilon <-  sample $ rvar (MultivariateNormal 0.0 bigR)
  let yNew = vector [sin x1New] + epsilon
  return $ Just ((xNew, yNew), xNew)

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

m0' :: R 2
m0' = vector [1.6, 0]

pendulumSamples' :: [(R 2, R 1)]
pendulumSamples' = evalState (ML.unfoldrM (pendulumSample bigQ' bigR') m0') (pureMT 17)

observe :: R 2 -> R 1
observe a = vector [sin x] where x = fst $ headTail a

linearizedObserve :: R 2 -> L 1 2
linearizedObserve a = matrix [cos x, 0.0] where x = fst $ headTail a

stateUpdate :: R 2 -> R 2
stateUpdate u =  vector [x1 + x2 * deltaT, x2 - g * (sin x1) * deltaT]
  where
    (x1, w) = headTail u
    (x2, _) = headTail w

linearizedStateUpdate :: R 2 -> Sq 2
linearizedStateUpdate u = matrix [1.0,                    deltaT,
                                  -g * (cos x1) * deltaT,    1.0]
  where
    (x1, w) = headTail u
    (x2, _) = headTail w

foo = runEKF (const observe) (const linearizedObserve) (const bigR')
             (const stateUpdate) (const linearizedStateUpdate) (const bigQ')
             undefined

baz = runEKFPrediction (const stateUpdate) (const linearizedStateUpdate) (const bigQ')
                       undefined initialDist

initialDist :: MultiNormal (R 2)
initialDist = MultiNormal (vector [0.0, 0.0])
                          (sym $ matrix [1.0, 0.0,
                                         0.0, 1.0])

bar = scanl foo initialDist (map (vector . pure) ws)
  where
    us = map fst pendulumSamples'
    vs = map (snd . headTail) us
    ws = map (fst . headTail) vs

test = take 10 bar

denv :: DEnv Double
denv = unsafePerformIO $ defaultEnv vectorAlignmentFns 600 500

diagEstimated :: String ->
                 [(Double, Double)] ->
                 [(Double, Double)] ->
                 [(Double, Double)] ->
                 Diagram Cairo
diagEstimated t l xs es =
  fst $ runBackend denv (render (chartEstimated t l xs es) (600, 500))

chartEstimated :: String ->
              [(Double, Double)] ->
              [(Double, Double)] ->
              [(Double, Double)] ->
              Renderable ()
chartEstimated title acts obs ests = toRenderable layout
  where

    actuals = plot_lines_values .~ [acts]
            $ plot_lines_style  . line_color .~ opaque red
            $ plot_lines_title .~ "Actual Trajectory"
            $ plot_lines_style  . line_width .~ 1.0
            $ def

    measurements = plot_points_values .~ obs
                 $ plot_points_style  . point_color .~ opaque blue
                 $ plot_points_title .~ "Measurements"
                 $ def

    estimas = plot_lines_values .~ [ests]
            $ plot_lines_style  . line_color .~ opaque black
            $ plot_lines_title .~ "Inferred Trajectory"
            $ plot_lines_style  . line_width .~ 1.0
            $ def

    layout = layout_title .~ title
           $ layout_plots .~ [toPlot actuals, toPlot measurements, toPlot estimas]
           $ layout_y_axis . laxis_title .~ "Angle / Horizontal Displacement"
           $ layout_y_axis . laxis_override .~ axisGridHide
           $ layout_x_axis . laxis_title .~ "Time"
           $ layout_x_axis . laxis_override .~ axisGridHide
           $ def

displayHeader :: FilePath -> Diagram B -> IO ()
displayHeader fn =
  mainRender ( DiagramOpts (Just 900) (Just 700) fn
             , DiagramLoopOpts False Nothing 0
             )

main = do
  let xs = take 1000 bar
  putStrLn $ show $ last xs
  let mus = map (fst . headTail . mu) xs
  let obs = take 1000 ws
        where
          us = map fst pendulumSamples'
          vs = map (snd . headTail) us
          ws = map (fst . headTail) vs
  let acts = take 1000 vs
        where
          us = map fst pendulumSamples'
          vs = map (fst . headTail) us
  displayHeader "diagrams/PendulumFittedEkf.png"
                (diagEstimated "Fitted Pendulum"
                               (zip [0,1..] acts)
                               (zip [0,1..] obs)
                               (zip [0,1..] mus))
