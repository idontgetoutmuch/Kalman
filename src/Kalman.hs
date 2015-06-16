{-# OPTIONS_GHC -Wall                     #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing  #-}
{-# OPTIONS_GHC -fno-warn-type-defaults   #-}
{-# OPTIONS_GHC -fno-warn-unused-do-bind  #-}
{-# OPTIONS_GHC -fno-warn-missing-methods #-}
{-# OPTIONS_GHC -fno-warn-orphans         #-}

{-# LANGUAGE DataKinds                    #-}
{-# LANGUAGE ScopedTypeVariables          #-}
{-# LANGUAGE TypeOperators                #-}
{-# LANGUAGE TypeFamilies                 #-}

-- | An extended Kalman filter. Note that this could be generalized
-- further. If you need a Kalman filter and have a state update matrix
-- \(A\) then you can just use @const bigA@ as the function which
-- returns the Jacobian.

module Kalman (
  extKalman
  )where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static hiding ( create )
import Data.Maybe ( fromJust )

-- | The model for the extended Kalman filter is given by
--
-- \[
-- \begin{aligned}
-- \boldsymbol{x}_i &= \boldsymbol{a}(\boldsymbol{x}_{i-1}) + \boldsymbol{\psi}_{i-1} \\
-- \boldsymbol{y}_i &= {H}\boldsymbol{x}_i + \boldsymbol{\upsilon}_i
-- \end{aligned}
-- \]
--
-- where
--
-- * \(\boldsymbol{a}\)\ is some non-linear vector-valued state update
-- function.
--
-- * \(\boldsymbol{\psi}_{i}\) are independent identically normally
-- distributed random variables with mean 0 representing the fact that
-- the state update is noisy: \(\boldsymbol{\psi}_{i} \sim {\cal{N}}(0,Q)\).
--
-- * \({H}\) is a matrix which represents how we observe the
-- hidden state process.
--
-- * \(\boldsymbol{\upsilon}_i\) are independent identically normally
-- distributed random variables with mean 0 represent the fact that
-- the observations are noisy: \(\boldsymbol{\upsilon}_{i} \sim {\cal{N}}(0,R)\).
--
-- We assume the whole process starts at 0 with our belief of the state (aka the
-- prior state) being given by
-- \(\boldsymbol{x}_0 \sim {\cal{N}}(\boldsymbol{\mu}_0, \Sigma_0)\)
extKalman ::  forall m n .
              (KnownNat m, KnownNat n, (1 <=? n) ~ 'True, (1 <=? m) ~ 'True) =>
              R n              -- ^ Prior mean \(\boldsymbol{\mu}_0\)
              -> Sq n          -- ^ Prior variance \(\Sigma_0\)
              -> L m n         -- ^ Observation map \(H\)
              -> Sq m          -- ^ Observation noise \(R\)
              -> (R n -> R n)  -- ^ State update function \(\boldsymbol{a}\)
              -> (R n -> Sq n) -- ^ A function which returns the
                               -- Jacobian of the state update
                               -- function at a given point
                               -- \(\frac{\partial \boldsymbol{a}}{\partial \boldsymbol{x}}\)
              -> Sq n          -- ^ State noise \(Q\)
              -> [R m]         -- ^ List of observations \(\boldsymbol{y}_i\)
              -> [(R n, Sq n)] -- ^ List of posterior means and variances \((\hat{\boldsymbol{x}}_i, \hat{\Sigma}_i)\)
extKalman muPrior sigmaPrior bigH bigSigmaY
  littleA bigABuilder bigSigmaX ys = result
  where
    result = scanl update (muPrior, sigmaPrior) ys

    update :: (R n, Sq n) -> R m -> (R n, Sq n)
    update (xHatFlat, bigSigmaHatFlat) y =
      (xHatFlatNew, bigSigmaHatFlatNew)
      where

        v :: R m
        v = y - (bigH #> xHatFlat)

        bigS :: Sq m
        bigS = bigH <> bigSigmaHatFlat <> (tr bigH) + bigSigmaY

        bigK :: L n m
        bigK = bigSigmaHatFlat <> (tr bigH) <> (inv bigS)

        xHat :: R n
        xHat = xHatFlat + bigK #> v

        bigSigmaHat :: Sq n
        bigSigmaHat = bigSigmaHatFlat - bigK <> bigS <> (tr bigK)

        bigA :: Sq n
        bigA = bigABuilder xHat

        xHatFlatNew :: R n
        xHatFlatNew = littleA xHat

        bigSigmaHatFlatNew :: Sq n
        bigSigmaHatFlatNew = bigA <> bigSigmaHat <> (tr bigA) + bigSigmaX

inv :: (KnownNat n, (1 <=? n) ~ 'True) => Sq n -> Sq n
inv m = fromJust $ linSolve m eye
