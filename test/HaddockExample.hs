{-# OPTIONS_GHC -Wall                   #-}
{-# OPTIONS_GHC -fno-warn-type-defaults #-}

{-# LANGUAGE DataKinds #-}

module HaddockExample ( main ) where

import           Numeric.LinearAlgebra.Static
                 ( R, vector, Sym,
                   headTail, matrix, sym
                 )
import           Data.Random.Distribution.MultiNormal
import           Numeric.Kalman
import           Numeric.LinearAlgebra.Static

import Graphics.Rendering.Chart hiding ( Vector )
import Graphics.Rendering.Chart.Backend.Diagrams
import Diagrams.Backend.Cairo.CmdLine
import Diagrams.Prelude hiding ( render, Renderable, sample )
import Diagrams.Backend.CmdLine

import Data.Csv
import System.IO hiding ( hGetContents )
import Data.ByteString.Lazy ( hGetContents )
import qualified Data.Vector as V

deltaT, g :: Double
deltaT = 0.01
g  = 9.81

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
    (x1, _) = headTail u

singleEKF :: MultiNormal (R 2) -> R 1 -> MultiNormal (R 2)
singleEKF = runEKF (const observe) (const linearizedObserve) (const bigR)
             (const stateUpdate) (const linearizedStateUpdate) (const bigQ)
             undefined

singleUKF :: MultiNormal (R 2) -> R 1 -> MultiNormal (R 2)
singleUKF = runUKF (const observe) (const bigR) (const stateUpdate) (const bigQ)
             undefined

initialDist :: MultiNormal (R 2)
initialDist = MultiNormal (vector [1.6, 0.0])
                          (sym $ matrix [0.1, 0.0,
                                         0.0, 0.1])

multiEKF :: [ℝ] -> [MultiNormal (R 2)]
multiEKF obs = scanl singleEKF initialDist (map (vector . pure) obs)

multiUKF :: [ℝ] -> [MultiNormal (R 2)]
multiUKF obs = scanl singleUKF initialDist (map (vector . pure) obs)

denv :: IO (DEnv Double)
denv = defaultEnv vectorAlignmentFns 600 500

diagEstimated :: String ->
                 [(Double, Double)] ->
                 [(Double, Double)] ->
                 [(Double, Double)] ->
                 IO (Diagram Cairo)
diagEstimated t l xs es = do
  env <- denv
  return $ fst $ runBackend env (render (chartEstimated t l xs es) (600, 500))

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

main :: IO ()
main = do
  h <- openFile "matlabRNGs.csv" ReadMode
  cs <- hGetContents h
  let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
  case df of
    Left _ -> error "Whatever"
    Right generatedSamples -> do
      let xs = take 500 (multiEKF $ V.toList $ V.map fst generatedSamples)
      let mus = map (fst . headTail . mu) xs
      let obs = V.toList $ V.map fst generatedSamples
      let acts = V.toList $ V.map snd generatedSamples
      de1 <- diagEstimated "Fitted Pendulum"
             (zip [0,1..] acts)
             (zip [0,1..] obs)
             (zip [0,1..] mus)
      displayHeader "diagrams/PendulumFittedEkf.png" de1
      let ys = take 500 (multiUKF $ V.toList $ V.map fst generatedSamples)
      let nus = map (fst . headTail . mu) ys
      de2 <- diagEstimated "Fitted Pendulum"
             (zip [0,1..] acts)
             (zip [0,1..] obs)
             (zip [0,1..] nus)
      displayHeader "diagrams/PendulumFittedUkf.png" de2
